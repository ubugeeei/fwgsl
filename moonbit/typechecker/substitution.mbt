///|
/// Substitution: mapping from type variables to types.
pub(all) struct Substitution {
  map : Map[TyVarId, Ty]
}

///|
pub fn Substitution::new() -> Substitution {
  Substitution::{ map: Map::new() }
}

///|
pub fn Substitution::insert(
  self : Substitution,
  var : TyVarId,
  ty : Ty
) -> Unit {
  self.map[var] = ty
}

///|
pub fn Substitution::lookup(self : Substitution, var : TyVarId) -> Ty? {
  self.map.get(var)
}

///|
/// Compose two substitutions: apply `self` first, then `other`.
pub fn Substitution::compose(
  self : Substitution,
  other : Substitution
) -> Substitution {
  let result = Substitution::new()
  self.map.each(fn(var, ty) { result.insert(var, ty.apply_subst(other)) })
  other.map.each(fn(var, ty) {
    if result.map.get(var) is None {
      result.insert(var, ty)
    }
  })
  result
}

///|
/// Type scheme (polymorphic type).
pub(all) struct Scheme {
  vars : Array[TyVarId]
  ty : Ty
}

///|
pub fn Scheme::mono(ty : Ty) -> Scheme {
  Scheme::{ vars: Array::new(), ty }
}

///|
pub fn Scheme::poly(vars : Array[TyVarId], ty : Ty) -> Scheme {
  Scheme::{ vars, ty }
}

///|
/// Type environment: a name -> Scheme mapping.
pub(all) struct TypeEnv {
  bindings : Map[String, Scheme]
}

///|
pub fn TypeEnv::new() -> TypeEnv {
  TypeEnv::{ bindings: Map::new() }
}

///|
pub fn TypeEnv::insert(self : TypeEnv, name : String, scheme : Scheme) -> Unit {
  self.bindings[name] = scheme
}

///|
pub fn TypeEnv::lookup(self : TypeEnv, name : String) -> Scheme? {
  self.bindings.get(name)
}

///|
/// Free variables in the environment (vars free in any binding's type but
/// not bound by that binding's scheme vars). Sorted and deduplicated.
pub fn TypeEnv::free_vars(self : TypeEnv) -> Array[TyVarId] {
  let vars : Array[TyVarId] = Array::new()
  self.bindings.each(fn(_, scheme) {
    for v in scheme.ty.free_vars() {
      if not(contains_var_id(scheme.vars, v)) {
        vars.push(v)
      }
    }
  })
  sort_and_dedup(vars)
  vars
}

///|
fn contains_var_id(arr : Array[TyVarId], v : TyVarId) -> Bool {
  for x in arr {
    if x == v {
      return true
    }
  }
  false
}

///|
pub fn TypeEnv::max_var_id(self : TypeEnv) -> TyVarId? {
  let mut current : TyVarId? = None
  self.bindings.each(fn(_, scheme) {
    for v in scheme.vars {
      current = Some(max_uint_opt(current, v))
    }
    for v in scheme.ty.free_vars() {
      current = Some(max_uint_opt(current, v))
    }
  })
  current
}

///|
fn max_uint_opt(current : TyVarId?, v : TyVarId) -> TyVarId {
  match current {
    Some(c) => if v > c { v } else { c }
    None => v
  }
}
