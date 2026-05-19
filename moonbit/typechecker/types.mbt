///|
/// Type system for fwgsl.
///
/// Provides type representations, substitution, unification, and
/// Hindley-Milner type inference infrastructure for the fwgsl compiler.

///|
/// Unique type variable identifier.
pub typealias UInt as TyVarId

///|
/// Type representation.
pub(all) enum Ty {
  Var(TyVarId)
  Con(String)
  App(Ty, Ty)
  Arrow(Ty, Ty)
  Forall(Array[String], Ty)
  Nat(UInt64)
  Error
} derive(Eq)

///|
pub fn Ty::i32() -> Ty {
  Con("I32")
}

///|
pub fn Ty::f32() -> Ty {
  Con("F32")
}

///|
pub fn Ty::u32() -> Ty {
  Con("U32")
}

///|
pub fn Ty::bool() -> Ty {
  Con("Bool")
}

///|
pub fn Ty::unit() -> Ty {
  Con("()")
}

///|
pub fn Ty::arrow(from : Ty, to : Ty) -> Ty {
  Arrow(from, to)
}

///|
pub fn Ty::app(func : Ty, arg : Ty) -> Ty {
  App(func, arg)
}

///|
/// Check whether this type contains a given type variable.
pub fn Ty::contains_var(self : Ty, var : TyVarId) -> Bool {
  match self {
    Var(v) => v == var
    Con(_) | Nat(_) | Error => false
    App(f, a) => f.contains_var(var) || a.contains_var(var)
    Arrow(a, b) => a.contains_var(var) || b.contains_var(var)
    Forall(_, body) => body.contains_var(var)
  }
}

///|
/// Apply a substitution.
pub fn Ty::apply_subst(self : Ty, subst : Substitution) -> Ty {
  match self {
    Var(v) =>
      match subst.lookup(v) {
        Some(t) =>
          match t {
            Var(v2) => if v2 == v { t } else { t.apply_subst(subst) }
            _ => t.apply_subst(subst)
          }
        None => self
      }
    Con(_) | Nat(_) | Error => self
    App(f, a) => App(f.apply_subst(subst), a.apply_subst(subst))
    Arrow(a, b) => Arrow(a.apply_subst(subst), b.apply_subst(subst))
    Forall(vars, body) => Forall(vars, body.apply_subst(subst))
  }
}

///|
/// Collect free type variables (sorted and deduplicated).
pub fn Ty::free_vars(self : Ty) -> Array[TyVarId] {
  let out : Array[TyVarId] = Array::new()
  collect_free_vars(self, out)
  sort_and_dedup(out)
  out
}

///|
fn collect_free_vars(ty : Ty, out : Array[TyVarId]) -> Unit {
  match ty {
    Var(v) => out.push(v)
    Con(_) | Nat(_) | Error => ()
    App(f, a) => {
      collect_free_vars(f, out)
      collect_free_vars(a, out)
    }
    Arrow(a, b) => {
      collect_free_vars(a, out)
      collect_free_vars(b, out)
    }
    Forall(_, body) => collect_free_vars(body, out)
  }
}

///|
fn sort_and_dedup(arr : Array[TyVarId]) -> Unit {
  // Simple insertion sort over a small array.
  let n = arr.length()
  let mut i = 1
  while i < n {
    let key = arr[i]
    let mut j = i - 1
    while j >= 0 && arr[j] > key {
      arr[j + 1] = arr[j]
      j = j - 1
    }
    arr[j + 1] = key
    i = i + 1
  }
  // Dedup in place.
  if n <= 1 {
    return
  }
  let mut write = 1
  let mut read = 1
  while read < n {
    if arr[read] != arr[write - 1] {
      arr[write] = arr[read]
      write = write + 1
    }
    read = read + 1
  }
  // Drop the tail.
  while arr.length() > write {
    let _ = arr.pop()
  }
}

///|
fn normalized_type_constructor_name(name : String) -> String? {
  match name {
    "Array" | "Tensor" | "Ten" => Some("Tensor")
    "Vec" | "Vector" => Some("Vec")
    "Mat" | "Matrix" => Some("Mat")
    "Sca" | "Scalar" => Some("Scalar")
    "Options" | "Option" => Some("Option")
    _ => None
  }
}

///|
/// Normalize surface aliases into canonical internal type constructors.
pub fn normalize_type_aliases(ty : Ty) -> Ty {
  match ty {
    Con(name) =>
      Con(
        match normalized_type_constructor_name(name) {
          Some(canonical) => canonical
          None => name
        },
      )
    App(f, arg) => {
      let f = normalize_type_aliases(f)
      let arg = normalize_type_aliases(arg)
      match f {
        Con(name) => if name == "Scalar" { arg } else { App(f, arg) }
        _ => App(f, arg)
      }
    }
    Arrow(a, b) => Arrow(normalize_type_aliases(a), normalize_type_aliases(b))
    Forall(vars, body) => Forall(vars, normalize_type_aliases(body))
    Var(_) | Nat(_) | Error => ty
  }
}

///|
pub fn tensor_ty(dim : Ty, elem : Ty) -> Ty {
  App(App(Con("Tensor"), dim), elem)
}

///|
pub fn vector_ty(dim : UInt64, elem : Ty) -> Ty {
  App(App(Con("Vec"), Nat(dim)), elem)
}

///|
pub fn matrix_ty(rows : UInt64, cols : UInt64, elem : Ty) -> Ty {
  App(App(App(Con("Mat"), Nat(rows)), Nat(cols)), elem)
}

///|
pub fn option_ty(inner : Ty) -> Ty {
  App(Con("Option"), inner)
}

///|
pub fn result_ty(inner : Ty) -> Ty {
  App(Con("Result"), inner)
}

///|
pub fn pair_ty(left : Ty, right : Ty) -> Ty {
  App(App(Con("Pair"), left), right)
}

///|
/// Human-readable rendering of a `Ty` matching the Rust crate's `Display`.
pub fn Ty::display(self : Ty) -> String {
  match self {
    Var(v) => "t" + v.to_string()
    Con(name) => name
    App(f, a) => "(" + f.display() + " " + a.display() + ")"
    Arrow(from, to) => "(" + from.display() + " -> " + to.display() + ")"
    Forall(vars, body) => {
      let mut s = "forall"
      for v in vars {
        s = s + " " + v
      }
      s + ". " + body.display()
    }
    Nat(n) => n.to_string()
    Error => "<error>"
  }
}

///|
pub impl Show for Ty with output(self, logger) {
  logger.write_string(self.display())
}
