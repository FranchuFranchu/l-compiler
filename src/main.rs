// this loads a file with:
// - explicit captures
// - explicit types
// - explicit joy :D

use std::{collections::BTreeMap, iter, process::exit, sync::Arc};

use cranelift::{
    codegen::{
        Context,
        ir::{self, FuncRef, Function, SigRef, immediates::Offset32},
    },
    prelude::{isa::CallConv, *},
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use serde_derive::{Deserialize, Serialize};

// A LangValue represents a single L value
#[derive(Debug)]
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
enum Type {
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
struct Captures(Vec<String>);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case", transparent)]
struct AnnotatedCaptures(Vec<AnnotatedVar>);

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Expression {
    Var(String),
    Unit,
    Pair(Box<Expression>, Box<Expression>),
    Left(Box<Expression>, Type),
    Right(Box<Expression>, Type),
    Bottom(Captures, Box<Command>),
    Par(Captures, AnnotatedVar, AnnotatedVar, Box<Command>),
    Match(
        Captures,
        AnnotatedVar,
        Box<Command>,
        AnnotatedVar,
        Box<Command>,
    ),
    #[serde(skip)]
    CompiledChannel(AnnotatedCaptures, FuncId, Type),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Command {
    Cut(Expression, Expression),
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
                iter.collect(),
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
        for i in 0..input_type.value_count() {
            signature.params.push(value_param.clone());
        }

        let func_id = module.declare_anonymous_function(&signature).unwrap();
        builder.func.signature = signature;

        let entry_block = builder.create_block();
        builder.switch_to_block(entry_block);
        builder.append_block_params_for_function_params(entry_block);
        builder.seal_block(entry_block);

        builder.declare_var(Variable::new(0), module.isa().pointer_type());
        builder.def_var(Variable::new(0), builder.block_params(entry_block)[1]);

        // Load captures
        let captures_buffer = builder.block_params(entry_block)[0];
        for var in captures.0 {
            let mut vals = vec![];
            for offset in (0..var.r#type.value_count()).map(|x| x * value_bytes as usize) {
                vals.push(builder.ins().load(
                    value_type,
                    MemFlags::trusted(),
                    captures_buffer,
                    offset as i32,
                ));
            }
            vars.insert(var.name, var.r#type.fill_from(&mut vals.into_iter()));
        }
        // TODO free captures

        let val =
            input_type.fill_from(&mut builder.block_params(entry_block)[2..].into_iter().cloned());
        let mut this = FunctionCompiler {
            module,
            builder,
            vars,
            value_type,
            alloc,
            free,
            func_id,
        };

        (this, val)
    }
    fn close(mut self) -> FuncId {
        let id = self.func_id;
        self.builder.ins().trap(TrapCode::user(100).unwrap());
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
                        args.push(self.builder.use_var(Variable::new(0)));
                        sig.params.push(AbiParam::new(self.value_type));
                        sig.params.push(AbiParam::new(self.value_type));
                        for i in arg.values() {
                            args.push(i);
                            sig.params.push(AbiParam::new(self.value_type));
                        }
                        let sig = self.builder.import_signature(sig);
                        self.builder.ins().call_indirect(sig, function, &args);
                    }
                    a => unreachable!("can't cut between two non-closures! [ :( ] {:?}", a),
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
            Expression::Bottom(captures, command) => todo!(),
            Expression::Par(captures, annotated_var, annotated_var1, command) => todo!(),
            Expression::Match(captures, annotated_var, command, annotated_var1, command1) => {
                todo!()
            }
            Expression::CompiledChannel(captures, func_id, input_type) => {
                let fun_ref = self
                    .module
                    .declare_func_in_func(*func_id, &mut self.builder.func);
                let allocator = self.builder.use_var(Variable::new(0));
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
                self.builder.def_var(Variable::new(0), allocator);

                let mut offset = 0;
                for i in &captures.0 {
                    let val = self.vars.remove(&i.name).unwrap();
                    for val in val.values() {
                        self.builder
                            .ins()
                            .store(MemFlags::trusted(), val, captures_p, 0);
                        offset += self.value_type.bytes();
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
            e @ (Expression::Par(..) | Expression::Match(..) | Expression::Bottom(..)) => {
                let (captures, func_id, t) = self.compile_closure_expression(e);
                *e = Expression::CompiledChannel(captures, func_id, t);
            }
            Expression::CompiledChannel(captures, func_id, _) => {}
            _ => {}
        }
    }
    fn compile_closure_expression(
        &mut self,
        e: &mut Expression,
    ) -> (AnnotatedCaptures, FuncId, Type) {
        match e {
            Expression::Par(captures, fst, snd, command) => {
                let captures = core::mem::take(captures);
                let fst_old = self.pre_ctx.insert(fst.name.clone(), fst.r#type.clone());
                let snd_old = self.pre_ctx.insert(snd.name.clone(), snd.r#type.clone());
                self.compile_command_children(command);
                if let Some(fst_old) = fst_old {
                    self.pre_ctx.insert(fst.name.clone(), fst_old);
                }
                if let Some(snd_old) = snd_old {
                    self.pre_ctx.insert(snd.name.clone(), snd_old);
                }

                let captures = captures.clone();
                let annotated_captures = captures.annotate(&mut self.pre_ctx);

                // start a new function
                let (mut f_comp, val) = FunctionCompiler::new(
                    self,
                    Type::Pair(Box::new(fst.r#type.clone()), Box::new(snd.r#type.clone())),
                    annotated_captures.clone(),
                );
                let LangValue::Pair(a, b) = val else {
                    unreachable!()
                };
                f_comp.vars.insert(fst.name.clone(), *a);
                f_comp.vars.insert(snd.name.clone(), *b);
                f_comp.compile_command(command);
                let id = f_comp.close();
                self.module.define_function(id, &mut self.ctx).unwrap();

                (
                    annotated_captures,
                    id,
                    Type::Dual(Box::new(Type::Pair(
                        Box::new(fst.r#type.clone()),
                        Box::new(snd.r#type.clone()),
                    ))),
                )
            }
            Expression::Match(captures, lft_name, lft_cmd, rgt_name, rgt_cmd) => todo!(),
            _ => unreachable!(),
        }
    }
    fn compile_command_children(&mut self, command: &mut Command) {
        match command {
            Command::Cut(a, b) => {
                self.compile_expression_children(a);
                self.compile_expression_children(b);
            }
        }
    }
}
extern "C" fn alloc(a: usize, b: usize) -> usize {
    0
}
pub fn main() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let mut builder = JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());
    builder.symbol("alloc", alloc as *const _);

    let mut module = JITModule::new(builder);

    let mut sig = Signature::new(isa::CallConv::SystemV);
    sig.params.push(AbiParam::new(isa.pointer_type()));
    sig.params.push(AbiParam::new(isa.pointer_type()));
    sig.returns.push(AbiParam::new(isa.pointer_type()));
    sig.returns.push(AbiParam::new(isa.pointer_type()));

    let alloc_func_id = module
        .declare_function("alloc", Linkage::Import, &sig)
        .unwrap();

    let free_func_id = module
        .declare_function("free", Linkage::Import, &sig)
        .unwrap();

    let int = ir::Type::int(64).unwrap();
    let mut ctx = Context::new();

    let mut compiler = Compiler {
        module,
        ctx,
        f_ctx: FunctionBuilderContext::new(),
        pre_ctx: Default::default(),
        alloc: Some(alloc_func_id),
        free: Some(free_func_id),
    };
    // (par () (" ") ((name . "b") (type . unit)) (cut (var . "a") (var . "b")))

    let mut closure: Result<Expression, serde_lexpr::Error> = serde_lexpr::from_str(
        "(par () (\"a\" unit) (\"b\" (dual . unit)) (cut (var . \"a\") (var . \"b\")))",
    );
    let Ok(mut closure) = closure else {
        let e = closure.unwrap_err();
        println!("{:?} {}", e.location(), e);
        todo!()
    }; /*
    let mut closure = Expression::Par(
    Captures(vec![]),
    ("a".to_string(), Type::Unit).into(),
    ("b".to_string(), Type::Unit).into(),
    Box::new(Command::Cut(
    Expression::Var("a".to_string()),
    Expression::Var("b".to_string()),
    )),
    );
    println!("{}", serde_lexpr::to_string(&closure).unwrap()); */
    let (_, func_id, _) = compiler.compile_closure_expression(&mut closure);

    type Allocator = i64;
    compiler.module.finalize_definitions().unwrap();
    let func = unsafe {
        std::mem::transmute::<
            _,
            extern "C" fn(
                *const u8,
                Allocator,
                *const u8, /*  extern "C" fn(*const u8, Allocator) */
                i64,
            ) -> !,
        >(compiler.module.get_finalized_function(func_id))
    };

    let mut our_captures: Box<String> = Box::new(String::from("hello world"));
    let our_captures = Box::into_raw(our_captures);
    extern "C" fn return_point(captures: *const u8, allocator: Allocator) {
        let captures: Box<String> = unsafe { Box::from_raw(captures as _) };
        println!("we got inside! captures: {captures:?} allocator: {allocator:?}");
        exit(0);
    }

    let mut their_captures: Box<[u8; 0]> = Box::new([]);
    let their_captures = Box::into_raw(their_captures);

    let allocator = 1234;
    func(
        their_captures as _,
        allocator,
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
