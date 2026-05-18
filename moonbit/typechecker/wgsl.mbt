///|
/// WGSL-compatible types (used in MIR and codegen).
pub(all) enum WgslType {
  I32
  U32
  F32
  Bool
  Vec(UInt, WgslType)
  Mat(UInt, UInt, WgslType)
  Struct(String)
  WgslArray(WgslType, UInt?)
  Unit
} derive(Eq)

///|
pub fn WgslType::display(self : WgslType) -> String {
  match self {
    I32 => "i32"
    U32 => "u32"
    F32 => "f32"
    Bool => "bool"
    Vec(n, elem) => "vec" + n.to_string() + "<" + elem.display() + ">"
    Mat(r, c, elem) =>
      "mat" + r.to_string() + "x" + c.to_string() + "<" + elem.display() + ">"
    Struct(name) => name
    WgslArray(elem, Some(n)) => "array<" + elem.display() + ", " + n.to_string() + ">"
    WgslArray(elem, None) => "array<" + elem.display() + ">"
    Unit => "void"
  }
}

///|
impl Show for WgslType with output(self : WgslType, logger : &Logger) -> Unit {
  logger.write_string(self.display())
}

///|
/// Convert a high-level `Ty` to a `WgslType` (after monomorphisation).
///
/// Returns `Err(String)` describing why the type cannot be lowered.
pub fn ty_to_wgsl(ty : Ty) -> Result[WgslType, String] {
  let ty = normalize_type_aliases(ty)
  match ty {
    Con(name) =>
      match name {
        "I32" => Ok(I32)
        "U32" => Ok(U32)
        "F32" => Ok(F32)
        "Bool" => Ok(Bool)
        "()" => Ok(Unit)
        _ => Ok(Struct(name))
      }
    App(f, arg) => app_to_wgsl(f, arg, ty)
    Arrow(_, _) => Err("Function types cannot be represented in WGSL")
    Var(v) => Err("Unresolved type variable t" + v.to_string())
    _ => Err("Cannot convert to WGSL: " + ty.display())
  }
}

///|
fn app_to_wgsl(f : Ty, arg : Ty, original : Ty) -> Result[WgslType, String] {
  match f {
    App(ff, n) =>
      match (ff, n) {
        // Mat N M Scalar
        (App(fff, nn), Nat(m)) =>
          match (fff, nn) {
            (Con(name), Nat(nv)) =>
              if name == "Mat" {
                let scalar = ty_to_wgsl(arg)
                match scalar {
                  Ok(elem) => Ok(Mat(nv.to_uint(), m.to_uint(), elem))
                  Err(e) => Err(e)
                }
              } else {
                Err("Cannot convert to WGSL: " + original.display())
              }
            _ => Err("Cannot convert to WGSL: " + original.display())
          }
        // Vec N Scalar
        (Con(name), Nat(nv)) =>
          if name == "Vec" {
            let scalar = ty_to_wgsl(arg)
            match scalar {
              Ok(elem) => Ok(Vec(nv.to_uint(), elem))
              Err(e) => Err(e)
            }
          } else if name == "Tensor" {
            let elem = ty_to_wgsl(arg)
            match elem {
              Ok(e) => Ok(WgslArray(e, Some(nv.to_uint())))
              Err(err) => Err(err)
            }
          } else {
            Err("Cannot convert to WGSL: " + original.display())
          }
        _ => Err("Cannot convert to WGSL: " + original.display())
      }
    Con(name) =>
      if name == "Tensor" {
        let elem = ty_to_wgsl(arg)
        match elem {
          Ok(e) => Ok(WgslArray(e, None))
          Err(err) => Err(err)
        }
      } else if name == "Vec" {
        Err("Partially applied Vec: " + original.display())
      } else {
        Err("Cannot convert to WGSL: " + original.display())
      }
    _ => Err("Cannot convert to WGSL: " + original.display())
  }
}
