; fwgsl indent queries for Zed

(function_declaration) @indent
(let_expression) @indent
(case_expression) @indent
(match_expression) @indent
(where_clause) @indent
(if_expression) @indent
(data_declaration) @indent
(trait_declaration) @indent
(impl_declaration) @indent

[")" "]" "}"] @outdent
"in" @outdent
