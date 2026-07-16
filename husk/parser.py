import os
from typing import List, Dict, Set
from tree_sitter import Parser, Node, Language
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

class CodeParser:
    """
    Parses source code files using tree-sitter to extract classes, functions, and import dependencies.
    """
    def __init__(self, language_name: str):
        self.language_name = language_name
        
        if language_name == "python":
            self.language = Language(tspython.language())
        elif language_name == "javascript":
            self.language = Language(tsjavascript.language())
        elif language_name == "typescript":
            try:
                self.language = Language(tstypescript.language_typescript())
            except AttributeError:
                self.language = Language(tstypescript.language())
        else:
            raise ValueError(f"Unsupported language: {language_name}")
            
        # Support tree-sitter version variations
        try:
            self.parser = Parser(self.language)
        except TypeError:
            self.parser = Parser()
            self.parser.set_language(self.language)

    def parse_file(self, file_path: str) -> Dict[str, any]:
        """
        Parses a file and returns extracted definitions and imports.
        """
        if not os.path.exists(file_path):
            return {"classes": [], "functions": [], "imports": []}
            
        with open(file_path, "rb") as f:
            code_bytes = f.read()
            
        tree = self.parser.parse(code_bytes)
        
        classes = []
        functions = []
        imports = set()
        
        self._traverse_node(tree.root_node, code_bytes, classes, functions, imports)
        
        return {
            "classes": classes,
            "functions": functions,
            "imports": sorted(list(imports)),
        }

    def _get_node_text(self, node: Node, code_bytes: bytes) -> str:
        """
        Extracts string content from a tree-sitter node.
        """
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

    def _traverse_node(
        self,
        node: Node,
        code_bytes: bytes,
        classes: List[Dict],
        functions: List[Dict],
        imports: Set[str]
    ):
        """
        Recursively traverses the AST nodes to extract relevant code structures.
        """
        node_type = node.type
        
        # Determine definitions based on language
        if self.language_name == "python":
            if node_type == "class_definition":
                name = self._get_name_attribute(node, code_bytes)
                classes.append({
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                })
            elif node_type == "function_definition":
                name = self._get_name_attribute(node, code_bytes)
                functions.append({
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                })
            elif node_type in ("import_statement", "import_from_statement"):
                self._extract_python_imports(node, code_bytes, imports)
                
        elif self.language_name in ("javascript", "typescript"):
            if node_type in ("class_declaration", "class"):
                name = self._get_name_attribute(node, code_bytes) or "AnonymousClass"
                classes.append({
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                })
            elif node_type in ("function_declaration", "method_definition", "function_expression", "arrow_function"):
                name = self._get_name_attribute(node, code_bytes)
                # For anonymous functions, only record if it's named via variable assignment
                if not name and node.parent and node.parent.type == "variable_declarator":
                    name = self._get_name_attribute(node.parent, code_bytes)
                
                if name:
                    functions.append({
                        "name": name,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                    })
            elif node_type == "import_statement":
                self._extract_js_ts_imports(node, code_bytes, imports)
            elif node_type == "call_expression":
                # Detect require('module') or import('module')
                self._extract_js_ts_requires(node, code_bytes, imports)

        # Recursively visit children
        for child in node.children:
            # Avoid traversing inside class/function definitions to find top-level function/classes,
            # but we still want to traverse inside them to find imports or nested functions if needed.
            # For simplicity, traverse all nodes.
            self._traverse_node(child, code_bytes, classes, functions, imports)

    def _get_name_attribute(self, node: Node, code_bytes: bytes) -> str:
        """
        Helper to get the name of a class, function, or variable declarator.
        """
        # Try tree-sitter field name first
        name_node = node.child_by_field_name("name")
        if name_node:
            return self._get_node_text(name_node, code_bytes)
            
        # Fallback to search child nodes by type 'identifier'
        for child in node.children:
            if child.type == "identifier":
                return self._get_node_text(child, code_bytes)
        return ""

    def _extract_python_imports(self, node: Node, code_bytes: bytes, imports: Set[str]):
        """
        Extracts imported module names from Python import AST nodes.
        """
        if node.type == "import_statement":
            # import os, sys -> children with type 'dotted_name' or 'aliased_import'
            for child in node.children:
                if child.type == "dotted_name":
                    imports.add(self._get_node_text(child, code_bytes))
                elif child.type == "aliased_import":
                    # import a.b as c -> dotted_name is the first child
                    dotted = child.child_by_field_name("name")
                    if dotted:
                        imports.add(self._get_node_text(dotted, code_bytes))
        elif node.type == "import_from_statement":
            # from a.b import c -> first dotted_name or relative_import is the module prefix
            module_name = ""
            for child in node.children:
                if child.type in ("dotted_name", "relative_import"):
                    module_name = self._get_node_text(child, code_bytes)
                    break
            if module_name:
                imports.add(module_name)

    def _extract_js_ts_imports(self, node: Node, code_bytes: bytes, imports: Set[str]):
        """
        Extracts import source paths from ES6 import statements.
        """
        # import defaultExport from "module-name";
        # import * as name from "module-name";
        source_node = node.child_by_field_name("source")
        if source_node:
            path = self._get_node_text(source_node, code_bytes).strip("\"'")
            imports.add(path)

    def _extract_js_ts_requires(self, node: Node, code_bytes: bytes, imports: Set[str]):
        """
        Extracts require() or dynamic import() arguments.
        """
        function_node = node.child_by_field_name("function")
        if function_node:
            func_name = self._get_node_text(function_node, code_bytes)
            if func_name in ("require", "import"):
                # Extract the first argument if it is a string
                arguments_node = node.child_by_field_name("arguments")
                if arguments_node and len(arguments_node.children) > 1:
                    first_arg = arguments_node.children[1] # index 0 is '('
                    if first_arg.type in ("string", "string_fragment"):
                        path = self._get_node_text(first_arg, code_bytes).strip("\"'`")
                        imports.add(path)

    def calculate_complexity(self, file_path: str) -> int:
        """
        Calculates syntactic complexity of a file based on AST decision points.
        """
        if not os.path.exists(file_path):
            return 0
            
        with open(file_path, "rb") as f:
            code_bytes = f.read()
            
        tree = self.parser.parse(code_bytes)
        complexity = [1] # baseline complexity of 1
        self._traverse_complexity(tree.root_node, code_bytes, complexity)
        return complexity[0]

    def _traverse_complexity(self, node: Node, code_bytes: bytes, complexity: List[int]):
        """
        Traverses nodes and increments complexity score for each decision/control point.
        """
        node_type = node.type
        
        if self.language_name == "python":
            if node_type in ("if_statement", "for_statement", "while_statement", "except_clause", "conditional_expression"):
                complexity[0] += 1
            elif node_type == "boolean_operator":
                complexity[0] += 1
        elif self.language_name in ("javascript", "typescript"):
            if node_type in (
                "if_statement", 
                "for_statement", 
                "for_in_statement", 
                "for_of_statement", 
                "while_statement", 
                "do_statement", 
                "catch_clause", 
                "ternary_expression",
                "switch_case",
                "case_clause"
            ):
                complexity[0] += 1
            elif node_type == "binary_expression":
                for child in node.children:
                    if child.type in ("&&", "||", "??"):
                        complexity[0] += 1
                        break
                        
        for child in node.children:
            self._traverse_complexity(child, code_bytes, complexity)

