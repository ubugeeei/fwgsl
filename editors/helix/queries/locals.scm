; fwgsl locals queries — scope and reference tracking

; Function declarations create a new scope
(function_declaration) @local.scope

; Let expressions create a new scope
(let_expression) @local.scope

; Where clauses create a new scope
(where_clause) @local.scope

; Lambda expressions create a new scope
(lambda_expression) @local.scope

; Case/match arms create a new scope
(case_arm) @local.scope
(match_arm) @local.scope

; Trait/impl declarations create a new scope
(trait_declaration) @local.scope
(impl_declaration) @local.scope

; Let binding name is a definition
(let_binding name: (identifier) @local.definition)

; Local binding name is a definition
(local_binding name: (identifier) @local.definition)

; Function declaration name is a definition
(function_declaration name: (identifier) @local.definition)

; Pattern identifiers are definitions
(identifier_pattern) @local.definition

; Identifier expressions are references
(identifier_expression) @local.reference
