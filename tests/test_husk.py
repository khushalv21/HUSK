import os
import shutil
import tempfile
import unittest
from husk.crawler import RepoCrawler
from husk.parser import CodeParser
from husk.graph import DependencyGraphBuilder

class TestHuskCore(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create python files
        self.py_main = os.path.join(self.test_dir, "main.py")
        with open(self.py_main, "w") as f:
            f.write("""
import os
from database import DBConnection
import utils

class Application:
    def __init__(self):
        self.db = DBConnection()

    def run(self):
        print("Running Application")

def helper_func():
    return True
""")

        self.py_db = os.path.join(self.test_dir, "database.py")
        with open(self.py_db, "w") as f:
            f.write("""
class DBConnection:
    def connect(self):
        pass
""")

        self.py_utils = os.path.join(self.test_dir, "utils.py")
        with open(self.py_utils, "w") as f:
            f.write("""
def parse_date(d):
    return d
""")

        # Create JavaScript / TypeScript files
        self.js_index = os.path.join(self.test_dir, "index.js")
        with open(self.js_index, "w") as f:
            f.write("""
const api = require('./api');
import { helper } from './helpers/utils';

class Server {
    start() {
        console.log("Server listening");
    }
}
""")

        self.js_api = os.path.join(self.test_dir, "api.js")
        with open(self.js_api, "w") as f:
            f.write("""
module.exports = {
    fetchData: function() { return 42; }
};
""")

        # Create ignored directories and files
        self.ignored_dir = os.path.join(self.test_dir, "node_modules")
        os.makedirs(self.ignored_dir)
        with open(os.path.join(self.ignored_dir, "leftover.js"), "w") as f:
            f.write("console.log('ignored');")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_crawler(self):
        crawler = RepoCrawler(self.test_dir)
        inventory = crawler.get_inventory()
        
        # Should find main.py, database.py, utils.py, index.js, api.js
        # and ignore node_modules/leftover.js
        files = {item["rel_path"] for item in inventory}
        self.assertIn("main.py", files)
        self.assertIn("database.py", files)
        self.assertIn("utils.py", files)
        self.assertIn("index.js", files)
        self.assertIn("api.js", files)
        self.assertNotIn("node_modules/leftover.js", files)
        self.assertEqual(len(files), 5)

    def test_python_parser(self):
        parser = CodeParser("python")
        result = parser.parse_file(self.py_main)
        
        # Test class extraction
        class_names = [c["name"] for c in result["classes"]]
        self.assertIn("Application", class_names)
        
        # Test function extraction
        func_names = [f["name"] for f in result["functions"]]
        self.assertIn("helper_func", func_names)
        self.assertIn("__init__", func_names)
        self.assertIn("run", func_names)
        
        # Test import extraction
        imports = result["imports"]
        self.assertIn("os", imports)
        self.assertIn("database", imports)
        self.assertIn("utils", imports)

    def test_js_parser(self):
        parser = CodeParser("javascript")
        result = parser.parse_file(self.js_index)
        
        # Test class extraction
        class_names = [c["name"] for c in result["classes"]]
        self.assertIn("Server", class_names)
        
        # Test function extraction
        func_names = [f["name"] for f in result["functions"]]
        self.assertIn("start", func_names)
        
        # Test import extraction (from ESM and CommonJS require)
        imports = result["imports"]
        self.assertIn("./api", imports)
        self.assertIn("./helpers/utils", imports)

    def test_dependency_graph(self):
        crawler = RepoCrawler(self.test_dir)
        inventory = crawler.get_inventory()
        
        parsed_data = {}
        for item in inventory:
            parser = CodeParser(item["language"])
            parsed_data[item["rel_path"]] = parser.parse_file(os.path.join(self.test_dir, item["rel_path"]))
            
        builder = DependencyGraphBuilder(inventory, parsed_data)
        graph = builder.build()
        
        # Test edges exist
        # main.py -> database.py (via database import)
        self.assertTrue(graph.has_edge("main.py", "database.py"))
        # main.py -> utils.py (via utils import)
        self.assertTrue(graph.has_edge("main.py", "utils.py"))
        # index.js -> api.js (via ./api import)
        self.assertTrue(graph.has_edge("index.js", "api.js"))
        
        # Test mermaid output contains correct node names and mappings
        mermaid_str = builder.to_mermaid()
        self.assertIn("main.py", mermaid_str)
        self.assertIn("database.py", mermaid_str)
        self.assertIn("index.js", mermaid_str)
        self.assertIn("api.js", mermaid_str)

    def test_complexity(self):
        parser = CodeParser("python")
        # Complex code has an if statement, a for loop, a boolean operator, so complexity > 1
        comp_file = os.path.join(self.test_dir, "complex.py")
        with open(comp_file, "w") as f:
            f.write("""
def check(x):
    if x > 10 and x < 20:
        for i in range(x):
            print(i)
""")
        complexity = parser.calculate_complexity(comp_file)
        # Baseline = 1, if = 2, and = 3, for = 4. Total = 4.
        self.assertEqual(complexity, 4)

    def test_git_analyzer(self):
        # Initialize a temporary git repository
        from git import Repo
        repo = Repo.init(self.test_dir)
        
        # Configure test git user
        with repo.config_writer() as writer:
            writer.set_value("user", "name", "Test User")
            writer.set_value("user", "email", "test@example.com")
            
        from husk.history import GitAnalyzer
        analyzer = GitAnalyzer(self.test_dir)
        self.assertTrue(analyzer.is_git_repo())
        
        # Create a file and commit it
        test_file = os.path.join(self.test_dir, "git_test.py")
        with open(test_file, "w") as f:
            f.write("print('git')")
            
        repo.index.add(["git_test.py"])
        repo.index.commit("Initial commit")
        
        metrics = analyzer.get_file_metrics("git_test.py")
        self.assertEqual(metrics["churn"], 1)
        self.assertEqual(metrics["authors"], 1)
        self.assertNotEqual(metrics["last_modified"], "Unknown")

    def test_file_cache(self):
        from husk.ai.cache import FileCache
        cache = FileCache(self.test_dir)
        
        test_file = os.path.join(self.test_dir, "cached_file.py")
        with open(test_file, "w") as f:
            f.write("content_v1")
            
        # Get missing
        self.assertIsNone(cache.get_summary("cached_file.py", test_file))
        
        # Set and retrieve
        cache.set_summary("cached_file.py", test_file, "This is a summary.")
        self.assertEqual(cache.get_summary("cached_file.py", test_file), "This is a summary.")
        
        # Modify file -> should cause a cache miss due to hash change
        with open(test_file, "w") as f:
            f.write("content_v2")
            
        self.assertIsNone(cache.get_summary("cached_file.py", test_file))

    def test_token_estimator(self):
        from husk.ai.estimator import TokenEstimator
        estimator = TokenEstimator("gpt-4o-mini")
        
        # Count tokens
        text = "Hello world, this is a test of the token estimator."
        tokens = estimator.count_tokens(text)
        self.assertGreater(tokens, 0)
        
        # Cost check
        cost, input_toks, output_toks = estimator.calculate_cost(1000)
        self.assertEqual(input_toks, 1000)
        self.assertEqual(output_toks, 250)
        
        # 1000 input at 0.15/1M + 250 output at 0.60/1M = 0.00015 + 0.00015 = 0.0003
        self.assertAlmostEqual(cost, 0.0003, places=6)
        
        formatted = estimator.format_cost(cost)
        self.assertEqual(formatted, "$0.00030")

    def test_summarizer_map_reduce(self):
        from husk.ai.summarizer import CodeSummarizer
        from husk.ai.adapters import OllamaAdapter
        from husk.ai.cache import FileCache
        
        adapter = OllamaAdapter("http://localhost:11434", "llama3")
        cache = FileCache(self.test_dir)
        summarizer = CodeSummarizer(self.test_dir, adapter, cache)
        
        inventory = [
            {"rel_path": "src/core/main.py", "language": "python", "size_bytes": 100},
            {"rel_path": "src/core/utils.py", "language": "python", "size_bytes": 50},
            {"rel_path": "src/db/conn.py", "language": "python", "size_bytes": 80},
            {"rel_path": "root_file.py", "language": "python", "size_bytes": 20},
        ]
        
        groups = summarizer.group_files_by_directory(inventory)
        self.assertEqual(len(groups), 3)
        self.assertIn("src/core", groups)
        self.assertIn("src/db", groups)
        self.assertIn(".", groups)
        
        self.assertEqual(set(groups["src/core"]), {"src/core/main.py", "src/core/utils.py"})
        self.assertEqual(groups["src/db"], ["src/db/conn.py"])
        self.assertEqual(groups["."], ["root_file.py"])
        
        os.makedirs(os.path.join(self.test_dir, "src/core"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "src/db"), exist_ok=True)
        
        file1 = os.path.join(self.test_dir, "src/core/main.py")
        file2 = os.path.join(self.test_dir, "src/core/utils.py")
        
        with open(file1, "w") as f:
            f.write("content1")
        with open(file2, "w") as f:
            f.write("content2")
            
        hash_v1 = summarizer.get_directory_combined_hash("src/core", groups["src/core"])
        self.assertTrue(len(hash_v1) > 0)
        
        self.assertIsNone(cache.get_dir_summary("src/core", hash_v1))
        cache.set_dir_summary("src/core", hash_v1, "Core logic module.")
        self.assertEqual(cache.get_dir_summary("src/core", hash_v1), "Core logic module.")
        
        with open(file1, "w") as f:
            f.write("content1_modified")
            
        hash_v2 = summarizer.get_directory_combined_hash("src/core", groups["src/core"])
        self.assertNotEqual(hash_v1, hash_v2)
        self.assertIsNone(cache.get_dir_summary("src/core", hash_v2))

    def test_cosine_similarity(self):
        from husk.ai.rag import VectorIndex
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        v4 = [1.0, 1.0, 0.0]
        
        self.assertAlmostEqual(VectorIndex._cosine_similarity(v1, v2), 1.0, places=5)
        self.assertAlmostEqual(VectorIndex._cosine_similarity(v1, v3), 0.0, places=5)
        self.assertAlmostEqual(VectorIndex._cosine_similarity(v1, v4), 0.707106, places=5)

    def test_syntax_aware_chunker(self):
        from husk.ai.rag import SyntaxAwareChunker
        content = """import os

class A:
    def __init__(self):
        pass

def global_func():
    return 1
"""
        parsed_data = {
            "classes": [{"name": "A", "start_line": 3, "end_line": 5}],
            "functions": [
                {"name": "__init__", "start_line": 4, "end_line": 5},
                {"name": "global_func", "start_line": 7, "end_line": 8}
            ]
        }
        
        chunks = SyntaxAwareChunker.chunk_file("test.py", content, parsed_data)
        types = [c["metadata"]["type"] for c in chunks]
        self.assertIn("class", types)
        self.assertIn("function", types)
        self.assertIn("module_level", types)
        
        class_chunk = [c for c in chunks if c["metadata"]["type"] == "class"][0]
        self.assertIn("class A:", class_chunk["text"])
        self.assertIn("pass", class_chunk["text"])

    def test_line_range_blame(self):
        from git import Repo
        repo = Repo.init(self.test_dir)
        
        with repo.config_writer() as writer:
            writer.set_value("user", "name", "Author Archaeologist")
            writer.set_value("user", "email", "arch@example.com")
            
        test_file = os.path.join(self.test_dir, "code.py")
        with open(test_file, "w") as f:
            f.write("def first():\n    pass\n\ndef second():\n    pass\n")
            
        repo.index.add(["code.py"])
        repo.index.commit("Initial version")
        
        with open(test_file, "w") as f:
            f.write("def first():\n    pass\n\ndef another_second():\n    print('why')\n")
            
        repo.index.add(["code.py"])
        repo.index.commit("Explain why")
        
        from husk.history import GitAnalyzer
        analyzer = GitAnalyzer(self.test_dir)
        
        blame1 = analyzer.get_line_range_blame("code.py", 1, 2)
        blame2 = analyzer.get_line_range_blame("code.py", 4, 5)
        
        self.assertNotEqual(blame1["sha"], "Unknown")
        self.assertNotEqual(blame2["sha"], "Unknown")
        self.assertEqual(blame2["message"], "Explain why")
        self.assertEqual(blame2["author"], "Author Archaeologist")

    def test_doc_generation_suite(self):
        from click.testing import CliRunner
        from husk.cli import doc
        from git import Repo
        
        try:
            Repo.init(self.test_dir)
        except Exception:
            pass
            
        runner = CliRunner()
        result = runner.invoke(doc, [self.test_dir])
        self.assertEqual(result.exit_code, 0)
        
        docs_path = os.path.join(self.test_dir, "docs")
        self.assertTrue(os.path.exists(os.path.join(docs_path, "index.md")))
        self.assertTrue(os.path.exists(os.path.join(docs_path, "dependency_graph.md")))
        self.assertTrue(os.path.exists(os.path.join(docs_path, "hotspots.md")))
        self.assertTrue(os.path.exists(os.path.join(docs_path, "deadcode.md")))

    def test_resolve_repo_path(self):
        from husk.cli import resolve_repo_path
        from unittest.mock import patch
        import tempfile
        import click
        
        path, temp_obj = resolve_repo_path(self.test_dir)
        self.assertEqual(path, os.path.abspath(self.test_dir))
        self.assertIsNone(temp_obj)
        
        with self.assertRaises(click.BadParameter):
            resolve_repo_path("/non/existent/path/for/husk/test")
            
        with patch("git.Repo.clone_from") as mock_clone:
            temp_path = os.path.join(self.test_dir, "temp_git_clone")
            os.makedirs(temp_path, exist_ok=True)
            
            with patch("tempfile.TemporaryDirectory") as mock_temp:
                mock_temp.return_value.name = temp_path
                
                url = "https://github.com/khushalv21/SYNTH.git"
                path, temp_obj = resolve_repo_path(url)
                
                self.assertEqual(path, temp_path)
                mock_clone.assert_called_once_with(url, temp_path)

    def test_handle_ai_error(self):
        from husk.cli import handle_ai_error
        
        e1 = Exception("Connection refused: could not connect")
        res1 = handle_ai_error(e1, "ollama", "llama3")
        self.assertIn("Please ensure the Ollama app is running", res1)
        
        e2 = Exception("model 'llama3' not found")
        res2 = handle_ai_error(e2, "ollama", "llama3")
        self.assertIn("ollama pull llama3", res2)
        
        e3 = Exception("insufficient_quota: you exceeded your quota")
        res3 = handle_ai_error(e3, "openai", "gpt-4o")
        self.assertIn("billing details", res3)
        
        e4 = Exception("Ollama request timed out: connection timed out")
        res4 = handle_ai_error(e4, "ollama", "llama3")
        self.assertIn("timed out", res4)

    def test_help_command(self):
        from click.testing import CliRunner
        from husk.cli import help
        
        runner = CliRunner()
        result = runner.invoke(help)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("HUSK CLI MENU", result.output)
        self.assertIn("COMMANDS:", result.output)
        self.assertIn("EXAMPLES:", result.output)

if __name__ == "__main__":
    unittest.main()
