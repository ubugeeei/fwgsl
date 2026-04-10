; shadml text object queries (for Helix mi/ma motions)

(function_declaration body: (_) @function.inside) @function.around
(data_declaration) @class.around
(trait_declaration) @class.around
(impl_declaration) @class.around
(type_signature) @class.around

(let_expression) @block.inside
(case_expression) @block.around
(match_expression) @block.around
(if_expression) @block.around

(line_comment) @comment.around
(block_comment) @comment.around

(record_fields) @parameter.inside
