#[derive(Debug)]
pub enum Error {
    InvalidSize,
}

type Result<T> = core::result::Result<T, Error>;

pub trait Abi {
    fn size() -> usize;
    unsafe fn deserialize(buf: &[usize]) -> Result<Self>
    where
        Self: Sized;
    fn serialize(&self, buf: &mut [usize]) -> Result<()>;
}

impl Abi for () {
    fn size() -> usize {
        0
    }

    unsafe fn deserialize(buf: &[usize]) -> Result<Self> {
        if buf.len() != 0 {
            return Err(Error::InvalidSize);
        }
        Ok(())
    }

    fn serialize(&self, buf: &mut [usize]) -> Result<()> {
        if buf.len() != 0 {
            return Err(Error::InvalidSize);
        }
        Ok(())
    }
}

impl<A: Abi, B: Abi> Abi for (A, B) {
    fn size() -> usize {
        A::size() + B::size()
    }

    unsafe fn deserialize(buf: &[usize]) -> Result<Self> {
        unsafe {
            if buf.len() != A::size() + B::size() {
                return Err(Error::InvalidSize);
            }
            Ok((
                A::deserialize(&buf[0..A::size()])?,
                B::deserialize(&buf[A::size()..])?,
            ))
        }
    }

    fn serialize(&self, buf: &mut [usize]) -> Result<()> {
        if buf.len() != 0 {
            return Err(Error::InvalidSize);
        }
        Ok(())
    }
}

impl<A: Abi, B: Abi> Abi for core::result::Result<A, B> {
    fn size() -> usize {
        1 + A::size().max(B::size())
    }

    unsafe fn deserialize(buf: &[usize]) -> Result<Self>
    where
        Self: Sized,
    {
        unsafe {
            if buf[0] != 0 {
                Ok(Err(B::deserialize(&buf[1..1 + B::size()])?))
            } else {
                Ok(Ok(A::deserialize(&buf[1..1 + A::size()])?))
            }
        }
    }

    fn serialize(&self, buf: &mut [usize]) -> Result<()> {
        match self {
            Ok(a) => {
                buf[0] = 0;
                a.serialize(&mut buf[1..1 + A::size()])?;
                Ok(())
            }

            Err(a) => {
                buf[0] = 1;
                a.serialize(&mut buf[1..1 + B::size()])?;
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
pub struct LangBox<T> {
    buffer: *mut usize,
    data: T,
}

impl<T: Abi> LangBox<T> {
    pub fn new(t: T) -> Self {
        let p = (crate::alloc(0, T::size() + 1) >> 64) as *mut usize;
        let buf = unsafe { &mut std::slice::from_raw_parts_mut(p, T::size() + 1)[1..] };
        t.serialize(buf).unwrap();
        unsafe { *p = 1 }
        Self { buffer: p, data: t }
    }
    fn dec_rc(&mut self) {
        unsafe {
            *self.buffer -= 1;
            if *self.buffer == 0 {
                crate::free(
                    0,
                    self.buffer as _,
                    (T::size() + 1) * (usize::BITS / 8) as usize,
                );
            }
        }
    }
    fn inc_rc(&mut self) {
        unsafe {
            *self.buffer += 1;
        }
    }
}

impl<T: Abi> Abi for LangBox<T> {
    fn size() -> usize {
        1
    }

    unsafe fn deserialize(buf: &[usize]) -> Result<Self>
    where
        Self: Sized,
    {
        if buf.len() != 1 {
            return Err(Error::InvalidSize);
        } else {
            let p = buf[0] as *const usize;

            let slice: &[usize] = unsafe { &std::slice::from_raw_parts(p, T::size() + 1)[1..] };
            Ok(Self {
                buffer: p as _,
                data: unsafe { T::deserialize(slice)? },
            })
        }
    }

    fn serialize(&self, buf: &mut [usize]) -> Result<()> {
        if buf.len() != 1 {
            return Err(Error::InvalidSize);
        } else {
            let slice =
                unsafe { &mut std::slice::from_raw_parts_mut(self.buffer, T::size() + 1)[1..] };
            self.data.serialize(slice)?;
            buf[0] = self.buffer as usize;
            Ok(())
        }
    }
}
