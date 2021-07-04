#![warn(clippy::all)]
#![allow(dead_code)]
#![feature(generic_associated_types)]
#![feature(maybe_uninit_extra)]

pub mod datalog;
pub mod query;
pub mod engine;
pub mod storage;
