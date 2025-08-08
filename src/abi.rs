use std::{
    collections::{HashMap, HashSet},
    sync::RwLock,
};

use cranelift_jit::JITModule;

use crate::Type;

pub const VALUE_BYTES: usize = (usize::BITS / 8) as usize;

#[cfg(target_pointer_width = "64")]
pub type TwoUsize = u128;
#[cfg(target_pointer_width = "32")]
pub type TwoUsize = u64;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct CallbackId(i32);

pub fn assemble_two_usize(a: usize, b: usize) -> TwoUsize {
    #[cfg(target_pointer_width = "64")]
    return a as u128 | (b as u128) << 64;
    #[cfg(target_pointer_width = "32")]
    return a as u64 | (b as u64) << 32;
}

pub fn disassemble_two_usize(n: TwoUsize) -> (usize, usize) {
    #[cfg(target_pointer_width = "64")]
    return ((n & usize::MAX as u128) as usize, (n >> 64) as usize);
    #[cfg(target_pointer_width = "32")]
    return ((n & usize::MAX as u64) as usize, (n >> 32) as usize);
}
pub extern "C" fn alloc_abi_compatible(mut allocator: usize, size: usize) -> TwoUsize {
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
pub extern "C" fn free_abi_compatible(allocator: usize, pointer: *mut usize, size: usize) -> usize {
    use std::alloc::{Layout, dealloc};
    println!("freed");
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe {
        dealloc(pointer as _, layout);
    }
    allocator
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct Value(pub usize);

pub struct VMCtxInner<'vm> {
    compiler: &'vm mut crate::Compiler,
    allocator: Option<Value>,
    /* allocator, callback_id, output_buffer, size */
    callback_buffer: Option<Box<[Value; 1]>>,
    callback_buffers: HashMap<CallbackId, (usize, Box<[Value; 2]>)>,
    next_callback_id: CallbackId,
    // trampolines
    // entry trampolines: take in a closure and N-size arguments, and call the closure
    // exit trampolines: write M size input into a buffer, and return
}

impl<'vm> VMCtxInner<'vm> {
    pub fn new(compiler: &'vm mut crate::Compiler) -> RwLock<Self> {
        RwLock::new(Self {
            compiler,
            allocator: Some(Value(0)),
            callback_buffer: Some(Box::new([Value(0)])),
            callback_buffers: HashMap::new(),
            next_callback_id: CallbackId(1),
        })
    }
    fn free_buffer(&mut self, buf: VMBuffer) {
        self.allocator = Some(Value(free_abi_compatible(
            self.allocator.as_mut().unwrap().0,
            buf.0,
            buf.1,
        )));
    }
    fn alloc_buffer(&mut self, size: usize) -> VMBuffer {
        let n = alloc_abi_compatible(self.allocator.as_mut().unwrap().0, size);
        let (allocator, ptr) = disassemble_two_usize(n);
        self.allocator = Some(Value(allocator));
        VMBuffer(ptr as _, size)
    }
    fn filled_buffer(&mut self, data: impl IntoIterator<Item = Value>) -> VMBuffer {
        let data: Vec<_> = data.into_iter().collect();
        let buf = self.alloc_buffer(data.len() * VALUE_BYTES);
        for i in 0..data.len() {
            unsafe { *buf.0.add(i) = data[i].0 };
        }
        buf
    }
    fn enter(
        &mut self,
        addr: Value,
        closure_state: Value,
        args: &[Value],
    ) -> (CallbackId, Vec<Value>) {
        let allocator = self.allocator.take().unwrap();
        let f = self.compiler.entry_trampoline_finalized(args.len());
        unsafe {
            f(
                allocator.0,
                addr.0 as _,
                closure_state.0 as _,
                args.as_ptr() as _,
            )
        }
        // now, see where we returned from.
        let p = self.callback_buffer.as_ref().unwrap()[0].0 as *mut Value;
        let allocator = unsafe { p.add(0).as_ref().unwrap().clone() };
        let callback_id = unsafe { p.add(1).as_ref().unwrap().clone() };
        self.allocator = Some(allocator);
        let (len, _) = self
            .callback_buffers
            .remove(&CallbackId(callback_id.0 as i32))
            .unwrap();
        (
            CallbackId(callback_id.0 as i32),
            Vec::from(unsafe { &std::slice::from_raw_parts(p as _, len)[2..] }),
        )
    }
    fn setup_exit(&mut self, len: usize) -> (CallbackId, Value, Value) {
        if self.callback_buffer.is_none() {
            self.callback_buffer = Some(Box::new([Value(0)]));
        }
        let callback_id = self.next_callback_id.clone();
        let buf = Box::new([
            Value(self.callback_buffer.as_ref().unwrap().as_ptr() as _),
            Value(callback_id.0 as usize),
        ]);
        let buf_p = Value(buf.as_ptr() as _);
        self.next_callback_id.0 += 1;
        self.callback_buffers
            .insert(callback_id.clone(), (len + 2, buf));
        let func_addr = self.compiler.exit_trampoline_finalized(len);
        (callback_id, Value(func_addr), buf_p)
    }
}

#[derive(Clone, Copy)]
pub struct VMCtx<'vm> {
    inner: &'vm RwLock<VMCtxInner<'vm>>,
}

impl<'vm> VMCtx<'vm> {
    pub fn new(inner: &'vm RwLock<VMCtxInner<'vm>>) -> Self {
        Self { inner }
    }
}

#[derive(Clone)]
struct VMBuffer(*mut usize, usize);

// note: everything is unsafe here :)
pub struct RawHandle<'vm> {
    state: VMCtx<'vm>,
    data: Vec<Value>,
    t: Type,
}

impl<'vm> RawHandle<'vm> {
    pub unsafe fn new(ctx: VMCtx<'vm>, data: Vec<Value>, t: Type) -> Self {
        Self {
            state: ctx,
            data,
            t,
        }
    }
    pub fn r#continue(&mut self) {
        let Type::Unit = core::mem::replace(&mut self.t, Type::Unit) else {
            unreachable!()
        };
    }
    pub fn receive(&mut self) -> RawHandle<'vm> {
        let Type::Pair(a, b) = core::mem::replace(&mut self.t, Type::Zero) else {
            unreachable!()
        };
        let new_data = self.data.split_off(a.value_count());
        self.t = *b;
        return RawHandle {
            state: self.state.clone(),
            data: new_data,
            t: *a,
        };
    }
    pub fn r#match(&mut self) -> bool {
        let Type::Either(a, b) = core::mem::replace(&mut self.t, Type::Zero) else {
            unreachable!()
        };
        let q = self.data[0].clone();
        if q.0 != 0 {
            self.data = Vec::from(&self.data[1..1 + b.value_count()]);
            self.t = *b;
            true
        } else {
            self.data = Vec::from(&self.data[1..1 + a.value_count()]);
            self.t = *a;
            false
        }
    }

    pub fn unit(ctx: VMCtx<'vm>) -> RawHandle<'vm> {
        return RawHandle {
            state: ctx,
            data: vec![],
            t: Type::Unit,
        };
    }
    // prepends a handle to a handle
    pub fn sent(&mut self, mut elem: &mut RawHandle<'vm>) -> &mut Self {
        elem.data.append(&mut self.data);
        self.data = core::mem::take(&mut elem.data);
        self.t = Type::Pair(
            Box::new(core::mem::replace(&mut elem.t, Type::Unit)),
            Box::new(core::mem::replace(&mut self.t, Type::Unit)),
        );
        self
    }
    // prepends a signal to a handle
    pub fn signaled(&mut self, signal: bool, alt: Type) -> &mut Self {
        let mut old_data = core::mem::replace(
            &mut self.data,
            vec![if signal { Value(1) } else { Value(0) }],
        );
        let max_len = self.t.value_count().max(alt.value_count());
        if max_len > old_data.len() {
            old_data.extend(std::iter::repeat(Value(0)).take(max_len - old_data.len()));
        }
        self.data.append(&mut old_data);
        if signal {
            self.t = Type::Pair(
                Box::new(alt),
                Box::new(core::mem::replace(&mut self.t, Type::Unit)),
            );
        } else {
            self.t = Type::Pair(
                Box::new(core::mem::replace(&mut self.t, Type::Unit)),
                Box::new(alt),
            );
        }
        self
    }
    // these are two kind-of dual functions

    pub unsafe fn cut(self, other: RawHandle) -> (CallbackId, Vec<Value>) {
        // use the VM context to enter into `self` with input `Handle`
        let Type::Dual(input_type) = self.t else {
            unreachable!()
        };
        let mut lock = self.state.inner.write().unwrap();
        let addr = self.data[0].clone();
        let closure = self.data[1].clone();
        lock.enter(addr, closure, &other.data)
    }

    pub unsafe fn callback(
        ctx: VMCtx,
        t: Type,
    ) -> (
        RawHandle,
        CallbackId, /* used later to identify what triggered the callback */
    ) {
        let mut lock = ctx.inner.write().unwrap();
        let (callback_id, func_addr, func_closure) = lock.setup_exit(t.value_count());
        (
            RawHandle {
                state: ctx,
                data: vec![func_addr, func_closure],
                t: Type::Dual(Box::new(t)),
            },
            callback_id,
        )

        // use the VM context to create a callback that returns back into the C ffi context, and allocates & fills in an output buffer.
        // note that all active callbacks share the same return buffer.
    }
}
