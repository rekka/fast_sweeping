// Copied from
// https://github.com/rust-lang/regex/blob/
// 37bfbd9d96676297372713dfe4a60afc5f6966a8/regex-capi/src/macros.rs
// under MIT license
macro_rules! ffi_fn {
    (fn $name:ident($($arg:ident: $arg_ty:ty),*,) -> $ret:ty $body:block) => {
        ffi_fn!(fn $name($($arg: $arg_ty),*) -> $ret $body);
    };
    (fn $name:ident($($arg:ident: $arg_ty:ty),*) -> $ret:ty $body:block) => {
        #[no_mangle]
        pub extern fn $name($($arg: $arg_ty),*) -> $ret {
            use ::std::io::{self, Write};
            use ::std::panic::{self, AssertUnwindSafe};
            use ::libc::abort;
            match panic::catch_unwind(AssertUnwindSafe(move || $body)) {
                Ok(v) => v,
                Err(err) => {
                    let msg = if let Some(&s) = err.downcast_ref::<&str>() {
                        s.to_owned()
                    } else if let Some(s) = err.downcast_ref::<String>() {
                        s.to_owned()
                    } else {
                        "UNABLE TO SHOW RESULT OF PANIC.".to_owned()
                    };
                    let _ = writeln!(
                        &mut io::stderr(),
                        "panic unwind caught, aborting: {:?}",
                        msg);
                    unsafe { abort() }
                }
            }
        }
    };
    (fn $name:ident($($arg:ident: $arg_ty:ty),*,) $body:block) => {
        ffi_fn!(fn $name($($arg: $arg_ty),*) -> () $body);
    };
    (fn $name:ident($($arg:ident: $arg_ty:ty),*) $body:block) => {
        ffi_fn!(fn $name($($arg: $arg_ty),*) -> () $body);
    };
}

macro_rules! sign_dist_ffi_fn {
    (fn $name:ident($ni:ident, $($nj:ident),*) -> $ret:ty) => {
        ffi_fn! {
            fn $name(d: *mut $ret, u: *const $ret,
                     $ni: size_t, $($nj: size_t,)* h: $ret) {
                let $ni = $ni as usize;
                $(let $nj = $nj as usize;)*
                let len = $ni $(* $nj)*;

                let d = unsafe { slice::from_raw_parts_mut(d, len) };
                let u = unsafe { slice::from_raw_parts(u, len) };

                fast_sweeping::$name(d, u, ($ni, $($nj, )*), h);
            }
        }
    }
}

macro_rules! haus_dist_ffi_fn {
    (fn $name:ident($ni:ident, $($nj:ident),*) -> $ret:ty) => {
        ffi_fn! {
            fn $name(u: *const $ret, v: *const $ret,
                     $ni: size_t, $($nj: size_t,)* h: $ret) -> $ret {
                let $ni = $ni as usize;
                $(let $nj = $nj as usize;)*
                let len = $ni $(* $nj)*;

                let u = unsafe { slice::from_raw_parts(u, len) };
                let v = unsafe { slice::from_raw_parts(v, len) };

                fast_sweeping::dist::$name(u, v, ($ni, $($nj, )*), h)
            }
        }
    }
}
