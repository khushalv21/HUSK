from dataclasses import dataclass, field
from typing import Optional, Set

@dataclass
class LanguageSpec:
    """
    Describes the tree-sitter node shapes CodeParser needs to know about for a language,
    so adding a new language means adding a spec entry instead of a new if/elif arm
    scattered across every traversal method.
    """
    # Node types that mark a class-like definition (class, interface, enum, ...)
    class_node_types: Set[str]
    # Node types that mark a function/method definition
    function_node_types: Set[str]
    # Node types that mark an import/require statement, dispatched to `import_extractor`
    import_node_types: Set[str]
    # Name of the CodeParser method (as a string) that extracts import targets for this language
    import_extractor: str
    # Node types that unconditionally add 1 to cyclomatic-style complexity
    complexity_node_types: Set[str]
    # If a class definition has no discoverable name, fall back to "AnonymousClass"
    # instead of leaving it blank (JS/TS allow anonymous class expressions; Python/Java don't).
    class_anonymous_fallback: bool = False
    # If a function definition has no discoverable name, try the enclosing
    # variable_declarator's name instead (`const foo = function() {}` style).
    function_variable_declarator_fallback: bool = False
    # If true, a function with no name at all (after any fallback) is skipped rather
    # than recorded with an empty name.
    function_requires_name: bool = False
    # Node type to inspect for short-circuit boolean operators (e.g. "binary_expression").
    # None means this language has a dedicated node type for boolean ops instead (see
    # complexity_node_types), so no child-token inspection is needed.
    conditional_binary_node_type: Optional[str] = None
    # Operator tokens that count as complexity when found as a child of conditional_binary_node_type
    boolean_operators: Set[str] = field(default_factory=set)


LANGUAGE_SPECS = {
    "python": LanguageSpec(
        class_node_types={"class_definition"},
        function_node_types={"function_definition"},
        import_node_types={"import_statement", "import_from_statement"},
        import_extractor="_extract_python_imports",
        complexity_node_types={
            "if_statement", "for_statement", "while_statement",
            "except_clause", "conditional_expression", "boolean_operator",
        },
    ),
    "javascript": LanguageSpec(
        class_node_types={"class_declaration", "class"},
        function_node_types={"function_declaration", "method_definition", "function_expression", "arrow_function"},
        import_node_types={"import_statement", "call_expression"},
        import_extractor="_extract_js_ts_imports_and_requires",
        complexity_node_types={
            "if_statement", "for_statement", "for_in_statement", "for_of_statement",
            "while_statement", "do_statement", "catch_clause", "ternary_expression",
            "switch_case", "case_clause",
        },
        class_anonymous_fallback=True,
        function_variable_declarator_fallback=True,
        function_requires_name=True,
        conditional_binary_node_type="binary_expression",
        boolean_operators={"&&", "||", "??"},
    ),
    "java": LanguageSpec(
        class_node_types={"class_declaration", "interface_declaration", "enum_declaration"},
        function_node_types={"method_declaration", "constructor_declaration"},
        import_node_types={"import_declaration"},
        import_extractor="_extract_java_imports",
        complexity_node_types={
            "if_statement", "for_statement", "enhanced_for_statement", "while_statement",
            "do_statement", "catch_clause", "ternary_expression", "switch_label",
        },
        conditional_binary_node_type="binary_expression",
        boolean_operators={"&&", "||"},
    ),
}
# TypeScript shares JavaScript's grammar shape for the node types we care about here.
LANGUAGE_SPECS["typescript"] = LANGUAGE_SPECS["javascript"]
