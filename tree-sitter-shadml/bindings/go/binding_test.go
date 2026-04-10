package tree_sitter_shadml_test

import (
	"testing"

	tree_sitter "github.com/smacker/go-tree-sitter"
	"github.com/tree-sitter/tree-sitter-shadml"
)

func TestCanLoadGrammar(t *testing.T) {
	language := tree_sitter.NewLanguage(tree_sitter_shadml.Language())
	if language == nil {
		t.Errorf("Error loading Shadmlsl grammar")
	}
}
