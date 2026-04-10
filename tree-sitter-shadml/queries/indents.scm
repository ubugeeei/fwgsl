; shadml indent queries

(function_declaration) @indent
(let_expression) @indent
(case_expression) @indent
(match_expression) @indent
(where_clause) @indent
(if_expression) @indent
(data_declaration) @indent
(trait_declaration) @indent
(impl_declaration) @indent

(case_arm) @indent
(match_arm) @indent

[")" "]" "}"] @outdent
"in" @outdent
"else" @outdent
