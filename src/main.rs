// this loads a file with:
// - explicit captures
// - explicit types
// - explicit joy :D

use std::{collections::BTreeMap, process::exit};

use cranelift::{
    codegen::{
        Context,
        ir::{self, BlockArg, FuncRef},
    },
    prelude::{isa::CallConv, *},
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use serde_derive::{Deserialize, Serialize};

// A LangValue represents a single L value
#[derive(Debug, Clone)]
pub enum LangValue {
    Pair(Box<LangValue>, Box<LangValue>),
    Either(Value, Vec<Value>, Type, Type),
    Unit,
    Closure(Value, Value, Type),
    Heap(Value),
}

// Define the input data structure
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Type {
    Unit,
    Zero,
    Pair(Box<Type>, Box<Type>),
    Either(Box<Type>, Box<Type>),
    Dual(Box<Type>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case", from = "(String, Type)")]
struct AnnotatedVar {
    name: String,
    r#type: Type,
}
impl From<(String, Type)> for AnnotatedVar {
    fn from((name, r#type): (String, Type)) -> Self {
        AnnotatedVar { name, r#type }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case", transparent)]
pub struct Captures(Vec<String>);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case", transparent)]
pub struct AnnotatedCaptures(Vec<AnnotatedVar>);

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Expression {
    Var(String),
    Unit,
    Pair(Box<Expression>, Box<Expression>),
    Left(Box<Expression>, Type),
    Right(Box<Expression>, Type),
    Chan(Captures, AnnotatedVar, Box<Command>),
    #[serde(skip)]
    CompiledChannel(AnnotatedCaptures, FuncId, Type),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Command {
    Cut(Expression, Expression),
    Receive(String, String, Box<Command>),
    Match(String, Box<Command>, Box<Command>),
    Continue(String, Box<Command>),
}
impl Type {
    fn value_count(&self) -> usize {
        match self {
            Type::Unit => 0,
            Type::Zero => 0,
            Type::Pair(a, b) => a.value_count() + b.value_count(),
            Type::Either(a, b) => a.value_count().max(b.value_count()) + 1,
            Type::Dual(_) => 2,
        }
    }
    fn fill_from(&self, iter: &mut impl Iterator<Item = Value>) -> LangValue {
        match self {
            Type::Unit => LangValue::Unit,
            Type::Zero => unreachable!(),
            Type::Pair(a, b) => {
                LangValue::Pair(Box::new(a.fill_from(iter)), Box::new(b.fill_from(iter)))
            }
            Type::Either(a, b) => LangValue::Either(
                iter.next().unwrap(),
                iter.by_ref()
                    .take(a.value_count().max(b.value_count()))
                    .collect(),
                a.as_ref().clone(),
                b.as_ref().clone(),
            ),
            Type::Dual(u) => LangValue::Closure(
                iter.next().unwrap(),
                iter.next().unwrap(),
                u.as_ref().clone(),
            ),
        }
    }
}

impl AnnotatedCaptures {
    fn value_count(&self) -> usize {
        self.0.iter().map(|x| x.r#type.value_count()).sum()
    }
}
impl LangValue {
    fn infer(&self) -> Type {
        match self {
            LangValue::Pair(a, b) => Type::Pair(Box::new(a.infer()), Box::new(b.infer())),
            LangValue::Either(value, values, a, b) => {
                Type::Either(Box::new(a.clone()), Box::new(b.clone()))
            }
            LangValue::Unit => Type::Unit,
            LangValue::Closure(_, _, t) => Type::Dual(Box::new(t.clone())),
            LangValue::Heap(value) => todo!(),
        }
    }
    fn values(&self) -> Vec<Value> {
        match self {
            LangValue::Pair(a, b) => [a.values(), b.values()].concat(),
            LangValue::Either(value, values, _, _) => {
                [vec![value.clone()], values.clone()].concat()
            }
            LangValue::Unit => vec![],
            LangValue::Closure(a, b, _) => vec![a.clone(), b.clone()],
            LangValue::Heap(value) => vec![value.clone()],
        }
    }
    fn fill_from(&mut self, iter: &mut impl Iterator<Item = Value>) {
        match self {
            LangValue::Pair(a, b) => {
                a.fill_from(iter);
                b.fill_from(iter);
            }
            LangValue::Either(value, values, ..) => {
                *value = iter.next().unwrap();
                for i in values.iter_mut() {
                    *i = iter.next().unwrap();
                }
            }
            LangValue::Unit => {}
            LangValue::Closure(a, b, _) => {
                *a = iter.next().unwrap();
                *b = iter.next().unwrap();
            }
            LangValue::Heap(value) => {
                *value = iter.next().unwrap();
            }
        }
    }
}

impl Captures {
    fn annotate(self, d: &BTreeMap<String, Type>) -> AnnotatedCaptures {
        AnnotatedCaptures(
            self.0
                .into_iter()
                .map(|x| AnnotatedVar {
                    name: x.clone(),
                    r#type: d.get(&x).unwrap().clone(),
                })
                .collect(),
        )
    }
}

pub struct Compiler {
    module: JITModule,
    ctx: Context,
    f_ctx: FunctionBuilderContext,
    pre_ctx: BTreeMap<String, Type>,
    alloc: Option<FuncId>,
    free: Option<FuncId>,
}

pub struct FunctionCompiler<'a> {
    module: &'a mut JITModule,
    builder: FunctionBuilder<'a>,
    vars: BTreeMap<String, LangValue>,
    value_type: ir::Type,
    alloc: FuncRef,
    free: FuncRef,
    func_id: FuncId,
    allocator: Variable,
}

impl<'a> FunctionCompiler<'a> {
    fn new(
        compiler: &'a mut Compiler,
        input_type: Type,
        captures: AnnotatedCaptures,
    ) -> (Self, LangValue) {
        let mut builder = FunctionBuilder::new(&mut compiler.ctx.func, &mut compiler.f_ctx);
        let module = &mut compiler.module;
        let mut signature = Signature::new(CallConv::SystemV);
        let value_type = module.isa().pointer_type();
        let value_bytes = module.isa().pointer_bytes();
        let value_param = AbiParam::new(module.isa().pointer_type());
        let mut vars = BTreeMap::new();

        let alloc = module.declare_func_in_func(compiler.alloc.unwrap(), builder.func);
        let free = module.declare_func_in_func(compiler.free.unwrap(), builder.func);
        // Captures
        signature.params.push(value_param.clone());
        // Allocator
        signature.params.push(value_param.clone());
        // Input
        for _ in 0..input_type.value_count() {
            signature.params.push(value_param.clone());
        }
        println!("chan: {:?}", signature);

        let func_id = module.declare_anonymous_function(&signature).unwrap();
        builder.func.signature = signature;

        let entry_block = builder.create_block();
        builder.switch_to_block(entry_block);
        builder.append_block_params_for_function_params(entry_block);
        builder.seal_block(entry_block);

        let allocator = builder.declare_var(module.isa().pointer_type());
        builder.def_var(allocator, builder.block_params(entry_block)[1]);

        // Load captures
        let captures_buffer = builder.block_params(entry_block)[0];
        let mut offset = 0;
        for var in captures.0 {
            let mut vals = vec![];
            for _ in 0..var.r#type.value_count() {
                vals.push(builder.ins().load(
                    value_type,
                    MemFlags::trusted(),
                    captures_buffer,
                    offset as i32,
                ));
                offset += value_bytes as usize;
            }
            vars.insert(var.name, var.r#type.fill_from(&mut vals.into_iter()));
        }

        // Emit free() call
        if offset > 0 {
            let length = builder.ins().iconst(value_type, offset as i64);
            let old_allocator = builder.use_var(allocator);
            let new_allocator = builder
                .ins()
                .call(free, &[old_allocator, captures_buffer, length]);
            let new_allocator = builder.inst_results(new_allocator)[0];
            builder.def_var(allocator, new_allocator);
        }

        let val =
            input_type.fill_from(&mut builder.block_params(entry_block)[2..].into_iter().cloned());
        let this = FunctionCompiler {
            module,
            builder,
            vars,
            value_type,
            alloc,
            free,
            func_id,
            allocator,
        };

        (this, val)
    }
    fn close(self) -> FuncId {
        let id = self.func_id;
        self.builder.finalize();
        id
    }
    fn compile_command(&mut self, command: &Command) {
        match command {
            Command::Cut(a, b) => {
                let a = self.compile_expression(a);
                let b = self.compile_expression(b);
                match (a, b) {
                    (LangValue::Closure(closure, function, t), arg)
                    | (arg, LangValue::Closure(closure, function, t)) => {
                        let mut sig = Signature::new(CallConv::SystemV);
                        let mut args = vec![];
                        args.push(closure);
                        args.push(self.builder.use_var(self.allocator));
                        sig.params.push(AbiParam::new(self.value_type));
                        sig.params.push(AbiParam::new(self.value_type));
                        for i in arg.values() {
                            args.push(i);
                            sig.params.push(AbiParam::new(self.value_type));
                        }
                        println!("cut: {:?}", sig);
                        let sig = self.builder.import_signature(sig);
                        self.builder.ins().call_indirect(sig, function, &args);
                        self.builder.ins().trap(TrapCode::user(30).unwrap());
                    }
                    a => unreachable!("can't cut between two non-closures! [ :( ] {:?}", a),
                }
            }
            Command::Continue(name, command) => {
                self.vars.remove(name);
                self.compile_command(command);
            }
            Command::Receive(name, dest, command) => {
                let expr = self.vars.remove(name).unwrap();
                let LangValue::Pair(a, b) = expr else {
                    unreachable!()
                };
                self.vars.insert(dest.clone(), *a);
                self.vars.insert(name.clone(), *b);
                self.compile_command(command);
            }
            Command::Match(name, lft, rgt) => {
                let expr = self.vars.remove(name).unwrap();
                let LangValue::Either(id, mut values, lft_t, rgt_t) = expr else {
                    unreachable!("{:?}", expr,);
                };
                let current_block = self.builder.current_block().unwrap();
                let lft_block = self.builder.create_block();
                let rgt_block = self.builder.create_block();

                let mut captured_values = vec![];
                for i in self.vars.iter_mut() {
                    captured_values.append(&mut i.1.values());
                }

                let mut passed_values = vec![self.builder.use_var(self.allocator)];
                let values_len = values.len();
                passed_values.append(&mut values);
                passed_values.append(&mut captured_values);
                for i in 0..passed_values.len() {
                    self.builder.append_block_param(lft_block, self.value_type);
                    self.builder.append_block_param(rgt_block, self.value_type);
                }

                let passed_values: Vec<_> = passed_values
                    .iter()
                    .map(|x| BlockArg::Value(x.clone()))
                    .collect();
                self.builder.ins().brif(
                    id,
                    rgt_block,
                    passed_values.iter(),
                    lft_block,
                    passed_values.iter(),
                );
                self.builder.seal_block(lft_block);
                self.builder.seal_block(rgt_block);

                let vars = std::mem::take(&mut self.vars);
                for (block, vars, input_type, command) in [
                    (lft_block, vars.clone(), lft_t, lft),
                    (rgt_block, vars, rgt_t, rgt),
                ] {
                    self.builder.switch_to_block(block);
                    let allocator = self.builder.block_params(block)[0];
                    let mut input_values = self.builder.block_params(block)[1..values_len + 1]
                        .iter()
                        .cloned();
                    let mut captured_values = self.builder.block_params(block)[values_len + 1..]
                        .iter()
                        .cloned();

                    self.vars = vars;
                    for i in self.vars.iter_mut() {
                        i.1.fill_from(&mut captured_values);
                    }
                    let input_value = input_type.fill_from(&mut input_values);
                    self.vars.insert(name.clone(), input_value);

                    self.builder.def_var(self.allocator, allocator);

                    self.compile_command(command);
                }
            }
        }
    }
    fn compile_expression_and_pad_to(
        &mut self,
        e: &Expression,
        t_b: Type,
    ) -> (Vec<Value>, Type, Type) {
        let e = self.compile_expression(e);
        let t_a = e.infer();
        let mut values = vec![];
        for _ in 0..(t_b.value_count() - t_a.value_count()).max(0) {
            values.push(self.builder.ins().iconst(self.value_type, 0))
        }
        (values, t_a, t_b)
    }
    fn compile_expression(&mut self, expression: &Expression) -> LangValue {
        match expression {
            Expression::Var(name) => self.vars.remove(name).unwrap(),
            Expression::Unit => LangValue::Unit,
            Expression::Pair(fst, snd) => LangValue::Pair(
                Box::new(self.compile_expression(fst)),
                Box::new(self.compile_expression(snd)),
            ),
            Expression::Left(e, other) => {
                let (values, t_a, t_b) = self.compile_expression_and_pad_to(e, other.clone());
                LangValue::Either(
                    self.builder.ins().iconst(self.value_type, 0),
                    values,
                    t_a,
                    t_b,
                )
            }
            Expression::Right(e, other) => {
                let (values, t_b, t_a) = self.compile_expression_and_pad_to(e, other.clone());
                LangValue::Either(
                    self.builder.ins().iconst(self.value_type, 1),
                    values,
                    t_a,
                    t_b,
                )
            }
            Expression::Chan(..) => todo!(),
            Expression::CompiledChannel(captures, func_id, input_type) => {
                let fun_ref = self
                    .module
                    .declare_func_in_func(*func_id, &mut self.builder.func);

                // emit alloc call
                let captures_p = if captures.value_count() > 0 {
                    let allocator = self.builder.use_var(self.allocator);
                    let size = captures
                        .0
                        .iter()
                        .map(|x| x.r#type.value_count())
                        .sum::<usize>()
                        * self.value_type.bytes() as usize;
                    let size = self.builder.ins().iconst(self.value_type, size as i64);
                    let call = self.builder.ins().call(self.alloc, &[allocator, size]);
                    let call = self.builder.inst_results(call);
                    let allocator = call[0];
                    let captures_p = call[1];
                    self.builder.def_var(self.allocator, allocator);
                    captures_p
                } else {
                    self.builder.ins().iconst(self.value_type, 0)
                };

                let mut offset = 0i32;
                for i in &captures.0 {
                    let val = self.vars.remove(&i.name).unwrap();
                    for val in val.values() {
                        self.builder
                            .ins()
                            .store(MemFlags::trusted(), val, captures_p, offset);
                        offset += self.value_type.bytes() as i32;
                    }
                }

                LangValue::Closure(
                    self.builder.ins().func_addr(self.value_type, fun_ref),
                    captures_p,
                    input_type.clone(),
                )
            }
        }
    }
}

// 3-split:
// function; context; level data.

impl Compiler {
    // These two functions compile all child closures of an expression or command
    fn compile_expression_children(&mut self, expression: &mut Expression) {
        match expression {
            Expression::Unit => {}
            Expression::Pair(a, b) => {
                self.compile_expression_children(a);
                self.compile_expression_children(b);
            }
            Expression::Left(a, _) => {
                self.compile_expression_children(a);
            }
            Expression::Right(b, _) => {
                self.compile_expression_children(b);
            }
            mut e @ Expression::Chan(..) => {
                let Expression::Chan(captures, var, command) = &mut e else {
                    unreachable!()
                };
                let fst_old = self.pre_ctx.insert(var.name.clone(), var.r#type.clone());
                self.compile_command_children(command);
                if let Some(fst_old) = fst_old {
                    self.pre_ctx.insert(var.name.clone(), fst_old);
                }

                let captures = core::mem::take(captures);
                let annotated_captures = captures.annotate(&mut self.pre_ctx);

                // start a new function
                let (mut f_comp, val) =
                    FunctionCompiler::new(self, var.r#type.clone(), annotated_captures.clone());
                f_comp.vars.insert(var.name.clone(), val);
                f_comp.compile_command(command);
                let func_id = f_comp.close();
                self.module.define_function(func_id, &mut self.ctx).unwrap();

                *e = Expression::CompiledChannel(
                    annotated_captures,
                    func_id,
                    Type::Dual(Box::new(var.r#type.clone())),
                );
            }
            Expression::CompiledChannel(captures, func_id, _) => {}
            _ => {}
        }
    }
    fn compile_command_children(&mut self, command: &mut Command) {
        match command {
            Command::Cut(a, b) => {
                self.compile_expression_children(a);
                self.compile_expression_children(b);
            }
            Command::Receive(_, _, command) => {
                self.compile_command_children(command);
            }
            Command::Match(_, lft, rgt) => {
                self.compile_command_children(lft);
                self.compile_command_children(rgt);
            }
            Command::Continue(_, command) => {
                self.compile_command_children(command);
            }
        }
    }

    fn internal_to_sysv<const N: u32, const M: u32>(
        &mut self,
        name: &str,
        // f: extern "C" fn(usize, *const u8, *const [usize; N], *const [MaybeUninit<usize>; M]) -> usize,
    ) -> FuncId {
        let pointer_type = self.module.isa().pointer_type();
        let mut sysv_signature = Signature::new(CallConv::SystemV);
        sysv_signature.params.push(AbiParam::new(pointer_type));
        sysv_signature.params.push(AbiParam::new(pointer_type));
        sysv_signature.params.push(AbiParam::new(pointer_type));

        // inner function signature
        let sysv_function_id = self
            .module
            .declare_function(name, Linkage::Import, &sysv_signature)
            .unwrap();

        let mut signature = Signature::new(CallConv::Tail);
        let pointer_type = self.module.isa().pointer_type();
        for i in 0..(N + 4) {
            signature.params.push(AbiParam::new(pointer_type));
        }
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.f_ctx);
        let func_id = self.module.declare_anonymous_function(&signature).unwrap();
        builder.func.signature = signature;
        let sysv_function_ref = self
            .module
            .declare_func_in_func(sysv_function_id, &mut builder.func);

        let entry_block = builder.create_block();
        let allocator = builder.block_params(entry_block)[0];
        let captures = builder.block_params(entry_block)[1];
        let ret_addr = builder.block_params(entry_block)[N as usize + 2];
        let ret_captures = builder.block_params(entry_block)[N as usize + 3];
        builder.switch_to_block(entry_block);
        builder.append_block_params_for_function_params(entry_block);
        builder.seal_block(entry_block);
        let in_slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            N * pointer_type.bytes(),
            4,
        ));
        let out_slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            M * pointer_type.bytes(),
            4,
        ));
        // copy input to stack
        for i in 0..N {
            let value = builder.block_params(entry_block)[i as usize + 2];
            builder
                .ins()
                .stack_store(value, in_slot, (i * pointer_type.bytes()) as i32);
        }

        let in_slot_addr = builder.ins().stack_addr(pointer_type, in_slot, 0);
        let out_slot_addr = builder.ins().stack_addr(pointer_type, out_slot, 0);
        let new_allocator = builder.ins().call(
            sysv_function_ref,
            &[allocator, captures, in_slot_addr, out_slot_addr],
        );

        let mut args = vec![builder.inst_results(new_allocator)[0], ret_captures];

        for i in 0..M {
            args.push(builder.ins().stack_load(
                pointer_type,
                out_slot,
                (i * pointer_type.bytes()) as i32,
            ));
        }

        func_id
    }
    /*
    fn sysv_to_internal<const N: u32>(&mut self, id: &FuncId) -> FuncId {

    } */
}

#[cfg(target_pointer_width = "64")]
pub type TwoUsize = u128;
#[cfg(target_pointer_width = "32")]
pub type TwoUsize = u64;

extern "C" fn alloc(allocator: usize, size: usize) -> TwoUsize {
    use std::alloc::{Layout, alloc};
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { alloc(layout) };

    #[cfg(target_pointer_width = "64")]
    return allocator as u128 | (ptr.expose_provenance() as u128) << 64;
    #[cfg(target_pointer_width = "32")]
    return allocator as u32 | (ptr.expose_provenance() as u32) << 64;
}
extern "C" fn free(allocator: usize, pointer: *mut u8, size: usize) -> usize {
    use std::alloc::{Layout, dealloc};
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe {
        dealloc(pointer, layout);
    }
    allocator
}
pub fn main() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    flag_builder.set("preserve_frame_pointers", "true").unwrap();
    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let mut builder = JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());
    builder.symbol("alloc", alloc as *const _);
    builder.symbol("free", free as *const _);

    let mut module = JITModule::new(builder);

    let mut sig = Signature::new(isa::CallConv::SystemV);
    sig.params.push(AbiParam::new(isa.pointer_type()));
    sig.params.push(AbiParam::new(isa.pointer_type()));
    sig.returns.push(AbiParam::new(isa.pointer_type()));
    sig.returns.push(AbiParam::new(isa.pointer_type()));

    let alloc_func_id = module
        .declare_function("alloc", Linkage::Import, &sig)
        .unwrap();

    let mut sig = Signature::new(isa::CallConv::SystemV);
    sig.params.push(AbiParam::new(isa.pointer_type()));
    sig.params.push(AbiParam::new(isa.pointer_type()));
    sig.params.push(AbiParam::new(isa.pointer_type()));
    sig.returns.push(AbiParam::new(isa.pointer_type()));
    let free_func_id = module
        .declare_function("free", Linkage::Import, &sig)
        .unwrap();

    let int = ir::Type::int(64).unwrap();
    let ctx = Context::new();

    let mut compiler = Compiler {
        module,
        ctx,
        f_ctx: FunctionBuilderContext::new(),
        pre_ctx: Default::default(),
        alloc: Some(alloc_func_id),
        free: Some(free_func_id),
    };
    // (par () (" ") ((name . "b") (type . unit)) (cut (var . "a") (var . "b")))

    let closure: Result<Expression, serde_lexpr::Error> = serde_lexpr::from_str(
        r#"
(chan () ("user" (pair (either unit unit) (pair (either unit unit) (dual . (either unit unit))))) (receive "user" "a" (receive "user" "b"
    (match "a"
        (continue "a"
            (match "b"
                (continue "b"
                    (cut (var . "user") (left unit unit))
                )
                (continue "b"
                    (cut (var . "user") (left unit unit))
                )
            )
        )
        (continue "a"
           (cut (var . "user") (var . "b"))
        )
    )
)))"#,
    );
    let Ok(mut closure) = closure else {
        let e = closure.unwrap_err();
        panic!("parsing error {:?} {}", e.location(), e);
    };
    /*
    let mut closure = Expression::Par(
    Captures(vec![]),
    ("a".to_string(), Type::Unit).into(),
    ("b".to_string(), Type::Unit).into(),
    Box::new(Command::Cut(
    Expression::Var("a".to_string()),
    Expression::Var("b".to_string()),
    )),
    );
    println!("{}", serde_lexpr::to_string(&closure).unwrap());
    */
    compiler.compile_expression_children(&mut closure);
    let Expression::CompiledChannel(captures, func_id, typ) = closure else {
        unreachable!()
    };

    type Allocator = i64;
    compiler.module.finalize_definitions().unwrap();
    let func = unsafe {
        std::mem::transmute::<_, extern "C" fn(*const u8, Allocator, i64, i64, *const u8, i64) -> !>(
            compiler.module.get_finalized_function(func_id),
        )
    };

    let our_captures: Box<String> = Box::new(String::from("hello world"));
    let our_captures = Box::into_raw(our_captures);
    extern "C" fn return_point(captures: *const u8, allocator: Allocator, e: usize) {
        println!(":)");
        let captures: Box<String> = unsafe { Box::from_raw(captures as _) };
        println!(
            "we got inside! captures: {captures:?} allocator: {allocator:?} number: {}",
            e
        );
        exit(0);
    }

    let their_captures: Box<[u8; 0]> = Box::new([]);
    let their_captures = Box::into_raw(their_captures);

    let allocator = 1234;
    func(
        their_captures as _,
        allocator,
        1,
        0,
        our_captures as _,
        return_point as _,
    );

    /*
    let fp_0 = unsafe {
    let fp_1 = unsafe {
        std::mem::transmute::<_, extern "C" fn(i64) -> i64>(
            module.get_finalized_function(func_id_1)
        )
    };
    println!("{}", unsafe { fp_0(fp_1(3), 6) })*/
}
