use cranelift::{codegen::Context, prelude::*};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

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
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    let int = Type::int(64).unwrap();
    let mut ctx = Context::new();

    ctx.func.signature.params.push(AbiParam::new(int));
    ctx.func.signature.params.push(AbiParam::new(int));
    ctx.func.signature.returns.push(AbiParam::new(int));
    let func_id = module
        .declare_function("multiply", Linkage::Export, &ctx.func.signature)
        .unwrap();

    let mut func_builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_builder_context);
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let a = builder.block_params(entry_block)[0];
    let b = builder.block_params(entry_block)[1];
    let ret = builder.ins().imul(a, b);
    builder.ins().return_(&[ret]);
    builder.finalize();
    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();

    let fp = unsafe {
        std::mem::transmute::<_, unsafe extern "C" fn(i64, i64) -> i64>(
            module.get_finalized_function(func_id),
        )
    };
    println!("{}", unsafe { (fp)(15, 16) })
}
