# Contributing to RAG from Scratch

First off, thank you for considering contributing to this project! 🎉

This is an educational project, and contributions that help others learn are especially welcome.

## 🎯 How Can I Contribute?

### 1. Report Bugs

If you find a bug, please create an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, Ollama version)

### 2. Suggest Enhancements

Have an idea? Open an issue tagged with `enhancement`:
- Describe the enhancement
- Explain why it would be useful for learning
- Provide examples if possible

### 3. Submit Code

**Good First Issues:**
- Add support for different file formats (PDF, DOCX)
- Implement alternative chunking strategies
- Add evaluation metrics
- Improve error messages
- Write additional documentation

**Process:**
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages: `git commit -m "feat: add PDF support"`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

## 📝 Code Style

- Follow PEP 8 for Python code
- Add comments explaining complex logic
- Keep functions small and focused
- Use descriptive variable names

**Example:**
```python
# Good
def calculate_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(vector_a, vector_b))
    # ... rest of implementation

# Avoid
def calc(a, b):
    dp = sum(x * y for x, y in zip(a, b))
    # ... rest of implementation
```

## 🧪 Testing

Before submitting:
1. Run the demo end-to-end
2. Test with different queries
3. Verify error handling works
4. Check that documentation is updated

## 📚 Documentation

If your contribution changes functionality:
- Update the README.md
- Add inline code comments
- Include usage examples

## 🎓 Educational Focus

Remember, this is a learning project! Contributions should:
- ✅ Be easy to understand
- ✅ Include explanatory comments
- ✅ Demonstrate concepts clearly
- ❌ Avoid over-engineering
- ❌ Don't add unnecessary complexity

## 💡 Ideas for Contributions

### Beginner-Friendly
- Add more example datasets
- Improve error messages
- Write tutorials for specific use cases
- Add logging for debugging

### Intermediate
- Implement semantic chunking
- Add support for multiple file formats
- Create a simple web interface
- Add evaluation metrics (precision, recall)

### Advanced
- Integrate vector database (ChromaDB, FAISS)
- Implement reranking
- Add multi-query retrieval
- Support conversation history
- Implement RAG fusion techniques

## 🤝 Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on learning and teaching
- Help others understand concepts

## ❓ Questions?

Feel free to:
- Open a discussion
- Comment on issues
- Reach out to maintainers

Thank you for contributing! 🙏
