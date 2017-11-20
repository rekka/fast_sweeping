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
