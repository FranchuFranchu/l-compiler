#![feature(box_patterns)]

// this loads a file with:
// - explicit captures
// - explicit types
// - explicit joy :D

pub mod parse;

use std::collections::{BTreeMap, HashMap};

use abi::{RawHandle, VMCtx, VMCtxInner, alloc_abi_compatible, free_abi_compatible};
use cranelift::{
    codegen::{
        Context,
        ir::{self, AbiParam, BlockArg, FuncRef, Signature},
    },
    prelude::{
        Configurable, FunctionBuilder, FunctionBuilderContext, InstBuilder, MemFlags,
        StackSlotData, StackSlotKind, Value, Variable, isa, isa::CallConv, settings,
    },
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use serde_derive::{Deserialize, Serialize};

pub mod abi;

pub type ContextBlueprint = BTreeMap<String, LangValue>;

// A LangValue represents a single L value
#[derive(Debug, Clone)]
pub enum LangValue {
    Pair(Box<LangValue>, Box<LangValue>),
    Either(Value, Vec<Value>, Type, Type),
    Unit,
    // func_addr, captures_buffer
    Closure(Value, Value, Type),
    // func_addr, captures_buffer
    // captures_buffer contains rc, buffer_len, and then the actual amount of captures
    BoxClosure(Value, Value, Type),
    // buffer
    Box(Value, Type),
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
    Box(Box<Type>),
    Dual(Box<Type>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case", from = "(String, Type)")]
pub struct AnnotatedVar {
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
    Box(Captures, Box<Expression>),
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
    Unbox(String, Box<Command>),
    Duplicate(String, String, Box<Command>),
    Erase(String, Box<Command>),
}
impl Type {
    fn value_count(&self) -> usize {
        match self {
            Type::Unit => 0,
            Type::Zero => 0,
            Type::Pair(a, b) => a.value_count() + b.value_count(),
            Type::Either(a, b) => a.value_count().max(b.value_count()) + 1,
            Type::Dual(_) => 2,
            Type::Box(_) => 1,
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
            LangValue::Either(_, _, a, b) => Type::Either(Box::new(a.clone()), Box::new(b.clone())),
            LangValue::Unit => Type::Unit,
            LangValue::Closure(_, _, t) => Type::Dual(Box::new(t.clone())),
            LangValue::BoxClosure(_, _, t) => Type::Box(Box::new(Type::Dual(Box::new(t.clone())))),
            LangValue::Heap(_) => todo!(),
            LangValue::Box(_, t) => t.clone(),
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
            LangValue::BoxClosure(a, b, _) => vec![a.clone(), b.clone()],
            LangValue::Heap(value) => vec![value.clone()],
            LangValue::Box(value, _) => vec![value.clone()],
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
            LangValue::BoxClosure(a, b, _) => {
                *a = iter.next().unwrap();
                *b = iter.next().unwrap();
            }
            LangValue::Heap(value) => {
                *value = iter.next().unwrap();
            }
            LangValue::Box(value, _) => {
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
    entry_trampolines: HashMap<usize, FuncId>,
    exit_trampolines: HashMap<usize, FuncId>,
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
        let mut signature = Signature::new(CallConv::Tail);
        let value_type = module.isa().pointer_type();
        let value_bytes = module.isa().pointer_bytes();
        let value_param = AbiParam::new(module.isa().pointer_type());
        let vars = BTreeMap::new();

        let alloc = module.declare_func_in_func(compiler.alloc.unwrap(), builder.func);
        let free = module.declare_func_in_func(compiler.free.unwrap(), builder.func);
        // Allocator
        signature.params.push(value_param.clone());
        // Captures
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
        builder.def_var(allocator, builder.block_params(entry_block)[0]);

        let mut this = FunctionCompiler {
            module,
            builder,
            vars,
            value_type,
            alloc,
            free,
            func_id,
            allocator,
        };

        // Load captures
        let captures_buffer = this.builder.block_params(entry_block)[1];
        let mut offset = 0;
        let value_count = captures.value_count();
        for var in captures.0 {
            let mut vals = vec![];
            for _ in 0..var.r#type.value_count() {
                vals.push(this.builder.ins().load(
                    value_type,
                    MemFlags::trusted(),
                    captures_buffer,
                    offset as i32,
                ));
                offset += value_bytes as usize;
            }
            let val = this.fill_from_type(
                &var.r#type,
                &mut ValueIterator::ValueList(vals.into_iter()),
                false,
            );
            this.vars.insert(var.name, val);
        }

        // Emit free() call
        this.emit_free_const(captures_buffer, value_count as i64);
        let it = Vec::from(&this.builder.block_params(entry_block)[2..]);

        let val = this.fill_from_type(
            &input_type,
            &mut ValueIterator::ValueList(it.into_iter()),
            false,
        );
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
                let a = self.compile_expression(a, false);
                let b = self.compile_expression(b, false);
                match (a, b) {
                    (LangValue::Closure(function, closure, t), arg)
                    | (arg, LangValue::Closure(function, closure, t)) => {
                        let mut sig = Signature::new(CallConv::Tail);
                        let mut args = vec![];
                        args.push(self.builder.use_var(self.allocator));
                        args.push(closure);
                        sig.params.push(AbiParam::new(self.value_type));
                        sig.params.push(AbiParam::new(self.value_type));
                        for i in arg.values() {
                            args.push(i);
                            sig.params.push(AbiParam::new(self.value_type));
                        }
                        println!("cut: {:?}", sig);
                        let sig = self.builder.import_signature(sig);
                        self.builder
                            .ins()
                            .return_call_indirect(sig, function, &args);
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
                let lft_block = self.builder.create_block();
                let rgt_block = self.builder.create_block();

                let (mut ctx, bp) = self.pack_context();

                ctx.append(&mut values);

                for _ in 0..ctx.len() {
                    self.builder.append_block_param(lft_block, self.value_type);
                }
                for _ in 0..ctx.len() {
                    self.builder.append_block_param(rgt_block, self.value_type);
                }

                let args: Vec<_> = ctx.iter().map(|x| BlockArg::Value(x.clone())).collect();
                self.builder
                    .ins()
                    .brif(id, rgt_block, args.iter(), lft_block, args.iter());
                self.builder.seal_block(lft_block);
                self.builder.seal_block(rgt_block);

                let vars = std::mem::take(&mut self.vars);
                for (block, vars, input_type, command) in [
                    (lft_block, vars.clone(), lft_t, lft),
                    (rgt_block, vars, rgt_t, rgt),
                ] {
                    self.builder.switch_to_block(block);
                    let args = Vec::from(self.builder.block_params(block));
                    let mut values = args.into_iter();
                    self.unpack_context(&mut values, &bp);
                    let input_value = self.fill_from_type(
                        &input_type,
                        &mut ValueIterator::ValueList(values.into_iter()),
                        false,
                    );
                    self.vars.insert(name.clone(), input_value);

                    self.compile_command(command);
                }
            }
            Command::Erase(name, command) => {
                let expr = self.vars.remove(name).unwrap();
                let LangValue::Box(mut buffer, t) = expr else {
                    unreachable!("{:?}", expr,);
                };
                let mut size = self
                    .builder
                    .ins()
                    .iconst(self.value_type, t.value_count() as i64 + 1);
                self.decrement_rc(&mut buffer, &mut size);
                self.compile_command(command);
            }
            Command::Unbox(name, command) => {
                let expr = self.vars.remove(name).unwrap();
                let LangValue::Box(mut buffer, t) = expr else {
                    unreachable!("{:?}", expr,);
                };
                let value = self.fill_from_type(
                    &t,
                    &mut ValueIterator::Buffer::<std::iter::Empty<_>>(
                        buffer,
                        self.value_type.bytes() as i32,
                    ),
                    true,
                );
                let mut size = self
                    .builder
                    .ins()
                    .iconst(self.value_type, t.value_count() as i64 + 1);
                self.vars.insert(name.clone(), value);
                self.decrement_rc(&mut buffer, &mut size);
                self.compile_command(command);
            }
            Command::Duplicate(name, name2, command) => {
                let expr = self.vars.remove(name).unwrap();
                let LangValue::Box(buffer, t) = expr else {
                    unreachable!("{:?}", expr,);
                };
                self.increment_rc(buffer);

                self.vars
                    .insert(name.clone(), LangValue::Box(buffer, t.clone()));
                self.vars.insert(name2.clone(), LangValue::Box(buffer, t));
                self.compile_command(command);
            }
        }
    }
    fn compile_expression_and_pad_to(
        &mut self,
        e: &Expression,
        t_b: Type,
        boxed: bool,
    ) -> (Vec<Value>, Type, Type) {
        let e = self.compile_expression(e, boxed);
        let t_a = e.infer();
        let mut values = vec![];
        for _ in 0..(t_b.value_count() - t_a.value_count()).max(0) {
            values.push(self.builder.ins().iconst(self.value_type, 0))
        }
        (values, t_a, t_b)
    }
    fn compile_expression(&mut self, expression: &Expression, boxed: bool) -> LangValue {
        match expression {
            Expression::Var(name) => self.vars.remove(name).unwrap(),
            Expression::Unit => LangValue::Unit,
            Expression::Pair(fst, snd) => LangValue::Pair(
                Box::new(self.compile_expression(fst, boxed)),
                Box::new(self.compile_expression(snd, boxed)),
            ),
            Expression::Left(e, other) => {
                let (values, t_a, t_b) =
                    self.compile_expression_and_pad_to(e, other.clone(), boxed);
                LangValue::Either(
                    self.builder.ins().iconst(self.value_type, 0),
                    values,
                    t_a,
                    t_b,
                )
            }
            Expression::Right(e, other) => {
                let (values, t_b, t_a) =
                    self.compile_expression_and_pad_to(e, other.clone(), boxed);
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
                let mut size = captures.value_count();

                if boxed {
                    size += 1
                }
                //let size = self.builder.ins().iconst(self.value_type, size as i64);
                let captures_p = self.emit_alloc_const(size as i64);
                let mut offset = 0i32;
                if boxed {
                    let initial_rc = self.builder.ins().iconst(self.value_type, 1);
                    self.builder
                        .ins()
                        .store(MemFlags::trusted(), initial_rc, captures_p, offset);
                    offset += self.value_type.bytes() as i32;
                }
                for i in &captures.0 {
                    let val = self.vars.remove(&i.name).unwrap();
                    for val in val.values() {
                        self.builder
                            .ins()
                            .store(MemFlags::trusted(), val, captures_p, offset);
                        offset += self.value_type.bytes() as i32;
                    }
                }

                if boxed {
                    LangValue::BoxClosure(
                        self.builder.ins().func_addr(self.value_type, fun_ref),
                        captures_p,
                        input_type.clone(),
                    )
                } else {
                    LangValue::Closure(
                        self.builder.ins().func_addr(self.value_type, fun_ref),
                        captures_p,
                        input_type.clone(),
                    )
                }
            }
            Expression::Box(captures, expr) => {
                let expr = self.compile_expression(expr, true);
                let t = expr.infer();
                let values = expr.values();
                let buffer = self.emit_alloc_const((values.len() + 1) as i64);
                let mut offset = 0i32;
                let initial_rc = self.builder.ins().iconst(self.value_type, 1);
                self.builder
                    .ins()
                    .store(MemFlags::trusted(), initial_rc, buffer, offset);
                offset += self.value_type.bytes() as i32;
                for i in values {
                    self.builder
                        .ins()
                        .store(MemFlags::trusted(), i, buffer, offset);
                    offset += self.value_type.bytes() as i32;
                }

                LangValue::Box(buffer, t)
            }
        }
    }
    fn emit_free_const(&mut self, pointer: Value, size: i64) {
        if size > 0 {
            let size = self.builder.ins().iconst(self.value_type, size);
            self.emit_free(pointer, size);
        }
    }
    fn emit_free(&mut self, pointer: Value, size: Value) {
        let m = self
            .builder
            .ins()
            .iconst(self.value_type, self.value_type.bytes() as i64);
        let size = self.builder.ins().imul(size, m);
        let old_allocator = self.builder.use_var(self.allocator);
        let new_allocator = self
            .builder
            .ins()
            .call(self.free, &[old_allocator, pointer, size]);
        let new_allocator = self.builder.inst_results(new_allocator)[0];
        self.builder.def_var(self.allocator, new_allocator);
    }
    fn emit_alloc_const(&mut self, size: i64) -> Value {
        if size > 0 {
            let size = self.builder.ins().iconst(self.value_type, size);
            self.emit_alloc(size)
        } else {
            self.builder.ins().iconst(self.value_type, 0)
        }
    }
    fn emit_alloc(&mut self, size: Value) -> Value {
        let s = self
            .builder
            .ins()
            .iconst(self.value_type, self.value_type.bytes() as i64);
        let size = self.builder.ins().imul(size, s);
        let allocator = self.builder.use_var(self.allocator);
        let call = self.builder.ins().call(self.alloc, &[allocator, size]);
        let call = self.builder.inst_results(call);
        let allocator = call[0];
        let buffer = call[1];
        self.builder.def_var(self.allocator, allocator);
        buffer
    }
    fn next_value<T: Iterator<Item = Value>>(&mut self, values: &mut ValueIterator<T>) -> Value {
        match values {
            ValueIterator::ValueList(i) => i.next().unwrap(),
            ValueIterator::Buffer(buffer, index) => {
                let v =
                    self.builder
                        .ins()
                        .load(self.value_type, MemFlags::trusted(), *buffer, *index);
                *index += self.value_type.bytes() as i32;
                v
            }
        }
    }
    fn fill_from_type<T: Iterator<Item = Value>>(
        &mut self,
        typ: &Type,
        values: &mut ValueIterator<T>,
        boxed: bool,
    ) -> LangValue {
        match typ {
            Type::Unit => LangValue::Unit,
            Type::Zero => unreachable!(),
            Type::Pair(a, b) => LangValue::Pair(
                Box::new(self.fill_from_type(a, values, boxed)),
                Box::new(self.fill_from_type(b, values, boxed)),
            ),
            Type::Either(a, b) => LangValue::Either(
                self.next_value(values),
                {
                    let mut v = vec![];
                    for _ in 0..(a.value_count().max(b.value_count())) {
                        v.push(self.next_value(values))
                    }
                    v
                },
                a.as_ref().clone(),
                b.as_ref().clone(),
            ),
            Type::Dual(u) => {
                if boxed {
                    LangValue::BoxClosure(
                        self.next_value(values),
                        self.next_value(values),
                        u.as_ref().clone(),
                    )
                } else {
                    LangValue::Closure(
                        self.next_value(values),
                        self.next_value(values),
                        u.as_ref().clone(),
                    )
                }
            }
            Type::Box(u) => LangValue::Box(self.next_value(values), u.as_ref().clone()),
        }
    }
    fn pack_context(&mut self) -> (Vec<Value>, ContextBlueprint) {
        let mut v = vec![self.builder.use_var(self.allocator)];
        for i in self.vars.iter_mut() {
            for value in i.1.values() {
                v.push(value)
            }
        }
        return (v, self.vars.clone());
    }
    fn unpack_context(&mut self, values: &mut impl Iterator<Item = Value>, bp: &ContextBlueprint) {
        self.builder.def_var(self.allocator, values.next().unwrap());
        let mut bp = bp.clone();
        for i in bp.iter_mut() {
            i.1.fill_from(values);
        }
        self.vars = bp
    }
    fn increment_rc(&mut self, buffer: Value) {
        // not atomic
        let rc = self
            .builder
            .ins()
            .load(self.value_type, MemFlags::trusted(), buffer, 0);
        let rc = self.builder.ins().iadd_imm(rc, 1);
        self.builder.ins().store(MemFlags::trusted(), rc, buffer, 0);
    }
    fn decrement_rc(&mut self, buffer_p: &mut Value, size_in_values_p: &mut Value) {
        let buffer = *buffer_p;
        let size_in_values = *size_in_values_p;
        // not atomic
        let rc = self
            .builder
            .ins()
            .load(self.value_type, MemFlags::trusted(), buffer, 0);
        let rc = self.builder.ins().iadd_imm(rc, -1);
        self.builder.ins().store(MemFlags::trusted(), rc, buffer, 0);

        let (mut ctx, bp) = self.pack_context();
        ctx.push(buffer);
        ctx.push(size_in_values);
        let ctx_args: Vec<_> = ctx.into_iter().map(|x| BlockArg::Value(x)).collect();
        let free_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        self.builder
            .ins()
            .brif(rc, else_block, ctx_args.iter(), free_block, ctx_args.iter());
        for _ in 0..ctx_args.len() {
            self.builder.append_block_param(free_block, self.value_type);
        }
        for _ in 0..ctx_args.len() {
            self.builder.append_block_param(else_block, self.value_type);
        }

        self.builder.seal_block(free_block);
        self.builder.switch_to_block(free_block);
        let params = Vec::from(self.builder.block_params(free_block));
        let mut params = params.into_iter();
        self.unpack_context(&mut params, &bp);
        let buffer = params.next().unwrap();
        let size_in_values = params.next().unwrap();

        self.emit_free(buffer, size_in_values);

        let (mut ctx, bp) = self.pack_context();
        ctx.push(buffer);
        ctx.push(size_in_values);
        let ctx_args: Vec<_> = ctx.into_iter().map(|x| BlockArg::Value(x)).collect();

        self.builder.ins().jump(else_block, ctx_args.iter());

        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);
        self.builder.switch_to_block(else_block);
        let params = Vec::from(self.builder.block_params(else_block));
        let mut params = params.into_iter();
        self.unpack_context(&mut params, &bp);
        let buffer = params.next().unwrap();
        let size_in_values = params.next().unwrap();
        *buffer_p = buffer;
        *size_in_values_p = size_in_values;
    }
}

enum ValueIterator<T: Iterator<Item = Value>> {
    ValueList(T),
    Buffer(Value, i32),
}

// 3-split:
// function; context; level data.

impl Compiler {
    // These two functions compile all child closures of an expression or command
    fn compile_expression_children(&mut self, expression: &mut Expression, boxed: bool) {
        match expression {
            Expression::Unit => {}
            Expression::Pair(a, b) => {
                self.compile_expression_children(a, boxed);
                self.compile_expression_children(b, boxed);
            }
            Expression::Left(a, _) => {
                self.compile_expression_children(a, boxed);
            }
            Expression::Right(b, _) => {
                self.compile_expression_children(b, boxed);
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

                println!("Channel:\n{}", &self.ctx.func);
                self.module.define_function(func_id, &mut self.ctx).unwrap();
                self.module.clear_context(&mut self.ctx);

                *e = Expression::CompiledChannel(
                    annotated_captures,
                    func_id,
                    Type::Dual(Box::new(var.r#type.clone())),
                );
            }
            Expression::CompiledChannel(..) => {}
            _ => {}
        }
    }
    fn compile_command_children(&mut self, command: &mut Command) {
        match command {
            Command::Cut(a, b) => {
                self.compile_expression_children(a, false);
                self.compile_expression_children(b, false);
            }
            Command::Match(_, lft, rgt) => {
                self.compile_command_children(lft);
                self.compile_command_children(rgt);
            }
            Command::Continue(_, command)
            | Command::Receive(_, _, command)
            | Command::Erase(_, command)
            | Command::Unbox(_, command)
            | Command::Duplicate(_, _, command) => {
                self.compile_command_children(command);
            }
        }
    }

    /// Generate an internal function with signature that system V abi function with symbol `name`.
    fn internal_to_sysv<const N: u32, const M: u32>(
        &mut self,
        name: &str,
        // f: extern "C" fn(usize, *const u8, *const [usize; N], *const [MaybeUninit<usize>; M]) -> usize,
    ) -> FuncId {
        // todo: probably doesn't work :)
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
        builder.switch_to_block(entry_block);
        builder.append_block_params_for_function_params(entry_block);
        builder.seal_block(entry_block);
        let allocator = builder.block_params(entry_block)[0];
        let captures = builder.block_params(entry_block)[1];
        let ret_addr = builder.block_params(entry_block)[N as usize + 2];
        let ret_captures = builder.block_params(entry_block)[N as usize + 3];
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

    // entry trampolines take in:
    // allocator, closure_addr, closure_state, pointer_to_args
    fn entry_trampoline(&mut self, n: usize) -> FuncId {
        let pointer_type = self.module.isa().pointer_type();

        if let Some(e) = self.entry_trampolines.get(&n) {
            return e.clone();
        }

        let mut target_signature = Signature::new(CallConv::Tail);
        for _ in 0..(n + 2) {
            target_signature.params.push(AbiParam::new(pointer_type));
        }

        let mut signature = Signature::new(CallConv::SystemV);
        // allocator
        // closure_addr
        // closure_state
        for _ in 0..(n + 3) {
            signature.params.push(AbiParam::new(pointer_type));
        }

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.f_ctx);
        let func_id = self.module.declare_anonymous_function(&signature).unwrap();
        builder.func.signature = signature;
        let sig_ref = builder.import_signature(target_signature);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let mut args = vec![
            builder.block_params(entry_block)[0],
            builder.block_params(entry_block)[2],
        ];
        let func_addr = builder.block_params(entry_block)[1];
        let input_buffer = builder.block_params(entry_block)[3];

        for i in 0..n {
            args.push(builder.ins().load(
                pointer_type,
                MemFlags::trusted(),
                input_buffer,
                (pointer_type.bytes() * i as u32) as i32,
            ))
        }
        builder.ins().call_indirect(sig_ref, func_addr, &args);
        builder.ins().return_(&[]);

        builder.finalize();

        println!("Entry trampoline for n = {n}:\n{}", &self.ctx.func);
        self.module.define_function(func_id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);

        self.entry_trampolines.insert(n, func_id);
        func_id
    }
    // exit trampolines take in:
    // allocator, closure_state ( output buffer ), args...
    fn exit_trampoline(&mut self, m: usize) -> FuncId {
        let pointer_type = self.module.isa().pointer_type();

        if let Some(e) = self.exit_trampolines.get(&m) {
            return e.clone();
        }

        let mut signature = Signature::new(CallConv::Tail);
        for _ in 0..(m + 2) {
            signature.params.push(AbiParam::new(pointer_type));
        }

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.f_ctx);
        let func_id = self.module.declare_anonymous_function(&signature).unwrap();
        builder.func.signature = signature;

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);
        let allocator = builder.block_params(entry_block)[0];
        let captures = builder.block_params(entry_block)[1];
        let output = Vec::from(&builder.block_params(entry_block)[2..]);
        let alloc = self
            .module
            .declare_func_in_func(self.alloc.unwrap(), builder.func);

        let active_buf = builder
            .ins()
            .load(pointer_type, MemFlags::trusted(), captures, 0);
        let callback_id = builder.ins().load(
            pointer_type,
            MemFlags::trusted(),
            captures,
            pointer_type.bytes() as i32,
        );

        let size = (m + 2) * pointer_type.bytes() as usize;
        let size = builder.ins().iconst(pointer_type, size as i64);
        let call = builder.ins().call(alloc, &[allocator, size]);
        let call = builder.inst_results(call);
        let allocator = call[0];
        let output_buffer = call[1];

        let mut offset = 0i32;
        builder
            .ins()
            .store(MemFlags::trusted(), allocator, output_buffer, offset);
        offset += pointer_type.bytes() as i32;
        builder
            .ins()
            .store(MemFlags::trusted(), callback_id, output_buffer, offset);
        offset += pointer_type.bytes() as i32;
        for i in 0..m {
            builder.ins().store(
                MemFlags::trusted(),
                output[i as usize],
                output_buffer,
                offset,
            );
            offset += pointer_type.bytes() as i32;
        }

        builder
            .ins()
            .store(MemFlags::trusted(), output_buffer, active_buf, 0);
        builder.ins().return_(&[]);

        builder.finalize();
        println!("Exit trampoline for m = {m}:\n{}", &self.ctx.func);
        self.module.define_function(func_id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);

        self.exit_trampolines.insert(m, func_id);
        func_id
    }

    fn entry_trampoline_finalized(
        &mut self,
        n: usize,
    ) -> unsafe extern "C" fn(usize, *const u8, *mut usize, *mut usize) {
        let tramp = self.entry_trampoline(n);
        self.module.finalize_definitions().unwrap();
        unsafe {
            core::mem::transmute::<_, unsafe extern "C" fn(usize, *const u8, *mut usize, *mut usize)>(
                self.module.get_finalized_function(tramp),
            )
        }
    }
    fn exit_trampoline_finalized(&mut self, m: usize) -> usize {
        let tramp = self.exit_trampoline(m);
        self.module.finalize_definitions().unwrap();
        self.module.get_finalized_function(tramp) as usize
    }
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
    builder.symbol("alloc", alloc_abi_compatible as *const _);
    builder.symbol("free", free_abi_compatible as *const _);

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

    let ctx = Context::new();

    let mut compiler = Compiler {
        module,
        ctx,
        f_ctx: FunctionBuilderContext::new(),
        pre_ctx: Default::default(),
        alloc: Some(alloc_func_id),
        free: Some(free_func_id),
        entry_trampolines: HashMap::new(),
        exit_trampolines: HashMap::new(),
    };
    // (par () (" ") ((name . "b") (type . unit)) (cut (var . "a") (var . "b")))

    let mut closure: Expression =
        Expression::from_lexpr(&lexpr::from_str(include_str!("../boolean.l")).unwrap()).unwrap();

    println!("{closure:#?}");

    compiler.compile_expression_children(&mut closure, true);
    let Expression::CompiledChannel(captures, func_id, typ) = closure else {
        unreachable!()
    };
    compiler.module.finalize_definitions().unwrap();
    let func_addr = compiler.module.get_finalized_function(func_id);

    let vm_ctx_inner = VMCtxInner::new(&mut compiler);
    let vm_ctx = VMCtx::new(&vm_ctx_inner);
    let h_type = Type::Dual(Box::new(typ.clone()));
    let Type::Dual(box Type::Pair(box a, box Type::Pair(box b, box Type::Dual(box c)))) = typ
    else {
        unreachable!()
    };

    for i in 0..4 {
        let captures = 0;
        let root_handle = unsafe {
            RawHandle::new(
                vm_ctx,
                vec![abi::Value(func_addr as usize), abi::Value(captures)],
                h_type.clone(),
            )
        };
        let (mut handle, cb) = unsafe { RawHandle::callback(vm_ctx, c.clone()) };
        let i0 = (i & 2) != 0;
        let i1 = (i & 1) != 0;
        handle.sent(RawHandle::unit(vm_ctx).signaled(i0, Type::Unit));
        handle.sent(RawHandle::unit(vm_ctx).signaled(i1, Type::Unit));
        let (r_cb, r_args) = unsafe { root_handle.cut(handle) };

        assert!(r_cb == cb); // we don't have any other callbacks
        let mut h = unsafe { RawHandle::new(vm_ctx, r_args, c.clone()) };
        let o = h.r#match();
        println!("f({i0}, {i1}) = {o}");
        h.r#continue();
    }
}
