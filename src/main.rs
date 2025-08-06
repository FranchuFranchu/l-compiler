// this loads a file with:
// - explicit captures
// - explicit types
// - explicit joy :D

use std::collections::BTreeMap;

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
            let val =
                this.fill_from_type(&var.r#type, &mut ValueIterator::ValueList(vals.into_iter()));
            this.vars.insert(var.name, val);
        }

        // Emit free() call
        this.emit_free_const(captures_buffer, value_count as i64);
        let it = Vec::from(&this.builder.block_params(entry_block)[2..]);

        let val = this.fill_from_type(&input_type, &mut ValueIterator::ValueList(it.into_iter()));
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
                let size = captures
                    .0
                    .iter()
                    .map(|x| x.r#type.value_count())
                    .sum::<usize>();
                //let size = self.builder.ins().iconst(self.value_type, size as i64);
                let captures_p = self.emit_alloc_const(size as i64);
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
            Expression::Box(captures, expr) => {
                let expr = self.compile_expression(expr);
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
    ) -> LangValue {
        match typ {
            Type::Unit => LangValue::Unit,
            Type::Zero => unreachable!(),
            Type::Pair(a, b) => LangValue::Pair(
                Box::new(self.fill_from_type(a, values)),
                Box::new(self.fill_from_type(b, values)),
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
            Type::Dual(u) => LangValue::Closure(
                self.next_value(values),
                self.next_value(values),
                u.as_ref().clone(),
            ),
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
                self.compile_expression_children(a);
                self.compile_expression_children(b);
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
    /// Generate a system V abi function that calls an VM function
    ///
    fn sysv_to_internal_entry_point<const N: u32, const M: u32>(
        &mut self,
        vm_id: &FuncId,
    ) -> FuncId {
        let pointer_type = self.module.isa().pointer_type();

        // Generate the return closure, which sets the return variables the output buffer and returns
        let mut ret_signature = Signature::new(CallConv::Tail);
        for _ in 0..(M + 2) {
            ret_signature.params.push(AbiParam::new(pointer_type));
        }

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.f_ctx);
        let ret_func_id = self
            .module
            .declare_anonymous_function(&ret_signature)
            .unwrap();
        builder.func.signature = ret_signature;

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);
        let allocator = builder.block_params(entry_block)[0];
        let captures = builder.block_params(entry_block)[1];
        let output = Vec::from(&builder.block_params(entry_block)[2..]);
        let mut offset = 0i32;
        builder
            .ins()
            .store(MemFlags::trusted(), allocator, captures, offset);

        offset += pointer_type.bytes() as i32;
        for i in 0..M {
            builder
                .ins()
                .store(MemFlags::trusted(), output[i as usize], captures, offset);
            offset += pointer_type.bytes() as i32;
        }
        builder.ins().return_(&[]);
        builder.finalize();
        println!("Ret trampoline:\n{}", &self.ctx.func);
        self.module
            .define_function(ret_func_id, &mut self.ctx)
            .unwrap();
        self.module.clear_context(&mut self.ctx);

        let mut signature = Signature::new(CallConv::SystemV);
        // allocator
        // captures
        // output buffer
        for _ in 0..(N + 3) {
            signature.params.push(AbiParam::new(pointer_type));
        }

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.f_ctx);
        let func_id = self.module.declare_anonymous_function(&signature).unwrap();
        builder.func.signature = signature;

        let ret_func_ref = self.module.declare_func_in_func(ret_func_id, builder.func);
        let vm_func_ref = self.module.declare_func_in_func(*vm_id, builder.func);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let mut args = vec![
            builder.block_params(entry_block)[0],
            builder.block_params(entry_block)[1],
        ];

        args.append(&mut Vec::from(&builder.block_params(entry_block)[3..]));
        args.push(builder.ins().func_addr(pointer_type, ret_func_ref)); // ret continuation address
        args.push(builder.block_params(entry_block)[2]); // output buffer = ret continuation captures

        builder.ins().call(vm_func_ref, &args);
        builder.ins().return_(&[]);

        builder.finalize();

        println!("Entry:\n{}", &self.ctx.func);
        self.module.define_function(func_id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);

        func_id
    }
}

#[cfg(target_pointer_width = "64")]
pub type TwoUsize = u128;
#[cfg(target_pointer_width = "32")]
pub type TwoUsize = u64;

extern "C" fn alloc(mut allocator: usize, size: usize) -> TwoUsize {
    // we don't use the allocator param right now
    // simply increment it
    use std::alloc::{Layout, alloc};
    let layout = Layout::from_size_align(size, 8).unwrap();
    let ptr = unsafe { alloc(layout) };
    allocator += 1;

    #[cfg(target_pointer_width = "64")]
    return allocator as u128 | (ptr.expose_provenance() as u128) << 64;
    #[cfg(target_pointer_width = "32")]
    return allocator as u32 | (ptr.expose_provenance() as u32) << 64;
}
extern "C" fn free(allocator: usize, pointer: *mut u8, size: usize) -> usize {
    use std::alloc::{Layout, dealloc};
    println!("freed");
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

    let closure: Result<Expression, serde_lexpr::Error> =
        serde_lexpr::from_str(include_str!("../dup3.l"));
    let Ok(mut closure) = closure else {
        let e = closure.unwrap_err();
        panic!("parsing error {:?} {}", e.location(), e);
    };
    compiler.compile_expression_children(&mut closure);
    let Expression::CompiledChannel(captures, func_id, typ) = closure else {
        unreachable!()
    };

    type Allocator = i64;
    let func_id = compiler.sysv_to_internal_entry_point::<1, 2>(&func_id);
    compiler.module.finalize_definitions().unwrap();

    let func = unsafe {
        // Allocator, their_captures, return_buffer, in0, in1
        std::mem::transmute::<_, extern "C" fn(Allocator, *const u8, *mut [i64; 3], i64)>(
            compiler.module.get_finalized_function(func_id),
        )
    };
    println!("{:p}", func);
    let allocator = 0;
    let mut input = alloc(0, 3 * 8) >> 64;
    unsafe { *(input as *mut [i64; 2]).as_mut().unwrap() = [1, 1] }

    let mut output_buffer: Box<[i64; 3]> = Box::new([1, 0, 0]);
    let output_buffer_p = output_buffer.as_mut() as *mut [i64; 3];
    func(
        allocator,
        output_buffer_p as *const u8,
        output_buffer_p as _,
        input as i64,
    );
    println!("{:?}", unsafe {
        (input as *mut [i64; 2]).as_mut().unwrap()
    });
    println!("{:?}", output_buffer);
}
