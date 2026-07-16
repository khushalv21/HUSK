# Contributing to Husk

Thank you for your interest in contributing to **Husk**! We welcome contributions of all forms, including bug reports, feature requests, documentation improvements, and pull requests.

---

## 🛠️ Development Setup

Husk requires Python >= 3.9. Follow these steps to set up a local development environment:

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/Husk.git
   cd Husk
   ```

2. **Initialize a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies and Editable Package:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

---

## 🧪 Testing

We maintain a comprehensive suite of unit tests. Before submitting a pull request, please ensure all tests pass:

```bash
# Run the entire test suite
python -m unittest tests/test_husk.py
```

When adding new features or modifying existing logic, please add corresponding unit test cases in `tests/test_husk.py`.

---

## 📝 Code Style & Guidelines

To maintain code quality, please adhere to the following standards:

1. **Formatting:** Follow standard PEP 8 formatting guidelines for Python code.
2. **Standard Library Preference:** Favor standard library solutions for AI adapters and HTTP requests to keep dependencies small and ensure portability on newer Python platforms (e.g., Python 3.13/3.14).
3. **No Placeholders:** Ensure code has complete error handling and robust defaults.

---

## 🚀 Pull Request Process

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/your-awesome-feature
   ```
2. Commit your changes and ensure tests are passing locally.
3. Push to your fork and submit a Pull Request (PR) to the `main` branch of the upstream repository.
4. Provide a clear description of the problem solved and the modifications made in your PR description.
