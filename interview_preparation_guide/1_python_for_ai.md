# Interview Preparation Guide, Module 1: Python for AI

Questions and answers for the Python foundations taught in Module 1. This is a study companion, not part of the published course site.

**A note on scope.** This guide is deliberately shorter than the others. It skips trivial syntax (what a `for` loop is) and focuses on the parts of Module 1 that actually come up in data and ML interviews: Python gotchas that trip people up, the data structures, and NumPy and Pandas. If you want the AI and ML concepts, start with guide 2 (ML and DL essentials); this one is the language and tooling floor beneath it.

**Coverage map.** The questions follow the five classes: 1.1 (Kickoff and setup), 1.2 (Python core: data structures, comprehensions, functions), 1.3 (Python for real work: OOP, errors, files, JSON), 1.4 (The data stack: NumPy and Pandas), and 1.5 (APIs and the web). Part 4 (NumPy and Pandas) is the most interviewed for data roles, so weight it.

## How to use this guide

Read the question, answer it out loud or on paper first, then expand the answer to check yourself. Each question is tagged by difficulty:

- **[Warm-up]** a definition or one-liner an interviewer opens with.
- **[Core]** the standard question you are expected to answer cleanly.
- **[Deep]** a mechanism or gotcha that separates strong candidates.
- **[Numerical]** a small trace to do by hand.
- **[Applied]** a scenario or short debugging prompt.

The gotchas in Parts 1 and 4 (mutability, default arguments, views vs copies, the Pandas copy trap) are the ones interviewers actually reach for.

---

## Part 1: Core Python and its gotchas (classes 1.1, 1.2)

### Q1. [Warm-up] List, tuple, set, dict: when do you reach for each?

<details><summary>Answer</summary>

List: an ordered, mutable sequence, for a collection you index and change. Tuple: an ordered, immutable sequence, for a fixed record you do not want mutated (and it can be a dict key). Set: an unordered collection of unique items, for membership tests and de-duplication, with fast `in`. Dict: key-to-value mapping, for lookups by name. Choose by the operation you need most: order and mutation (list), fixedness (tuple), uniqueness and membership (set), lookup by key (dict).
</details>

### Q2. [Core] What does mutable versus immutable mean, and which built-in types are which?

<details><summary>Answer</summary>

A mutable object can be changed in place after creation; an immutable one cannot, so "changing" it makes a new object. Immutable: int, float, str, bool, tuple, frozenset. Mutable: list, dict, set. It matters because two names can refer to the same mutable object, so a change through one is visible through the other, and because only immutable (hashable) objects can be dict keys or set members.
</details>

### Q3. [Deep] Why is a mutable default argument a bug? Show the classic case.

<details><summary>Answer</summary>

A default argument is evaluated once, when the function is defined, not on each call. So `def f(x, acc=[])` shares one list across all calls: appending to `acc` leaks into the next call. The fix is the sentinel pattern:

```python
def f(x, acc=None):
    if acc is None:
        acc = []
    acc.append(x)
    return acc
```

The same trap applies to `{}` and any mutable default.
</details>

### Q4. [Core] Difference between `is` and `==`?

<details><summary>Answer</summary>

`==` compares values (are these equal?); `is` compares identity (are these the exact same object in memory?). Use `==` for value checks and `is` only for singletons, chiefly `is None`. They can disagree: two equal lists are `==` but not `is`. They can accidentally agree for small ints and short strings because Python caches (interns) them, which is why relying on `is` for value comparison is a bug waiting to happen.
</details>

### Q5. [Deep] Distinguish a shallow copy from a deep copy.

<details><summary>Answer</summary>

A shallow copy (`list(x)`, `x[:]`, `copy.copy`) makes a new outer container but reuses the same inner objects, so mutating a nested element still affects both. A deep copy (`copy.deepcopy`) recursively copies everything, so the two are fully independent. For a list of lists, a shallow copy shares the inner lists; a deep copy does not. Assignment (`y = x`) copies nothing, it just binds another name to the same object.
</details>

### Q6. [Core] Explain list and dict comprehensions, and the two `if` positions.

<details><summary>Answer</summary>

A comprehension builds a collection in one expression: `[c*9/5+32 for c in celsius]`. A trailing `if` filters (`[x for x in xs if x > 0]`); a front `if/else` chooses the value (`["pass" if s >= 70 else "fail" for s in scores]`). Dict comprehension has the same shape building `key: value`. The rule of thumb from class: if you cannot read it aloud in one breath, use a loop instead.
</details>

### Q7. [Deep] What do `*args` and `**kwargs` do, and where does that matter later?

<details><summary>Answer</summary>

In a definition, `*args` gathers extra positional arguments into a tuple and `**kwargs` gathers extra keyword arguments into a dict; at a call site, `*` and `**` unpack a sequence or dict into arguments. It matters because it lets you forward options through a wrapper without naming them all, which is exactly how you pass extra parameters through to model and API calls later in the course (for example forwarding `**kwargs` to a `chat()` call).
</details>

### Q8. [Applied] `a = [1,2,3]; b = a; b.append(4)`. What is `a`, and how would you have avoided it?

<details><summary>Answer</summary>

`a` is `[1, 2, 3, 4]`, because `b = a` binds a second name to the same list, so the append is visible through both. To get an independent list, copy it: `b = a[:]` or `b = list(a)` (shallow), or `copy.deepcopy(a)` if it is nested. This is the mutability-plus-aliasing gotcha from Q2 and Q5 in practice.
</details>

---

## Part 2: OOP, errors, files, and JSON (class 1.3)

### Q9. [Warm-up] What is `self`, and what does `__init__` do?

<details><summary>Answer</summary>

`self` is the instance the method is called on, passed automatically as the first parameter, so a method can read and write that object's attributes. `__init__` is the initializer: it runs when you create an instance and sets up its starting attributes. The class is the cookie cutter; each instance is a cookie with its own state.
</details>

### Q10. [Core] What are inheritance and polymorphism, with `super()`?

<details><summary>Answer</summary>

Inheritance lets a subclass reuse and extend a base class; `super().__init__(...)` calls the base initializer so you do not repeat its setup, and a subclass can override a method to change behavior. Polymorphism means code that calls a method works across different types that implement it, so a loop over a mix of `Account` and `SavingsAccount` calls each one's own version without checking the type. Python also allows duck typing: if it has the method, it works.
</details>

### Q11. [Core] Why is a bare `except:` a bad habit?

<details><summary>Answer</summary>

A bare `except:` (or `except Exception`) swallows every error, including ones you did not anticipate (typos, keyboard interrupts, bugs), hiding failures and making debugging painful. Catch the specific exception you expect, for example `except ValueError:` or `except FileNotFoundError:`, so real bugs still surface. Handle what you can respond to; let the rest propagate.
</details>

### Q12. [Core] What does `with open(...)` give you over a plain `open`, and how does JSON map to Python?

<details><summary>Answer</summary>

`with open(...) as f:` is a context manager: it guarantees the file is closed when the block exits, even on an error, so you do not leak file handles. JSON maps directly onto Python types: a JSON object is a dict, an array is a list, and strings, numbers, booleans, and null map to `str`, `int`/`float`, `bool`, and `None`. `json.load`/`json.loads` parse into those types; `json.dump`/`json.dumps` serialize back. This is why an API's JSON response is just a dict once parsed.
</details>

### Q13. [Applied] Open a config file that might not exist and fall back to a default. Sketch it.

<details><summary>Answer</summary>

```python
import json
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}          # sensible default instead of crashing
```

Catch the specific `FileNotFoundError`, not everything, so a malformed-JSON error (a different failure) still surfaces rather than being silently swallowed.
</details>

---

## Part 3: NumPy (class 1.4)

### Q14. [Core] Why a NumPy array instead of a Python list for numeric work?

<details><summary>Answer</summary>

An array is a fixed-dtype, contiguous block of memory, so operations run as vectorized C loops instead of a slow Python-level loop, and it uses far less memory than a list of boxed Python objects. That gives elementwise math, broadcasting, and aggregations that are both faster and more concise. Lists are heterogeneous and flexible but not built for numeric compute.
</details>

### Q15. [Core] What is vectorization, and why avoid explicit loops?

<details><summary>Answer</summary>

Vectorization expresses an operation over a whole array at once (`a + b`, `a * 2`, `np.dot(a, b)`) so the loop runs in optimized C, not in the Python interpreter. It is faster and reads closer to the math. The habit to build: reach for a whole-array operation or a NumPy function before writing a `for` loop over elements.
</details>

### Q16. [Deep] Explain broadcasting and give the rule.

<details><summary>Answer</summary>

Broadcasting lets NumPy operate on arrays of different shapes without copying data, by virtually stretching the smaller one. The rule: compare shapes from the trailing (rightmost) dimension; two dimensions are compatible if they are equal or one of them is 1; a size-1 dimension is stretched to match. So a `(3, 4)` array plus a `(4,)` array works (the row is applied to each of the 3 rows); a `(3, 4)` plus `(3,)` fails unless you reshape to `(3, 1)`.
</details>

### Q17. [Deep] Does slicing a NumPy array copy it? Why does that matter?

<details><summary>Answer</summary>

No. A basic slice returns a view onto the same underlying data, so writing to the slice mutates the original array. That is efficient but a gotcha: `b = a[:2]; b[0] = 99` changes `a` too. Use `a[:2].copy()` when you need an independent array. (Fancy indexing with a list or boolean mask, by contrast, returns a copy.)
</details>

### Q18. [Numerical] For `a = np.array([[1,2,3],[4,5,6]])`, what are `a.sum(axis=0)` and `a.sum(axis=1)`?

<details><summary>Answer</summary>

`axis=0` collapses down the rows (sum each column): `[5, 7, 9]`. `axis=1` collapses across the columns (sum each row): `[6, 15]`. The mnemonic: `axis` is the dimension that disappears. `a.sum()` with no axis sums everything to `21`.
</details>

---

## Part 4: Pandas (class 1.4)

### Q19. [Warm-up] What is a Series versus a DataFrame?

<details><summary>Answer</summary>

A Series is a one-dimensional labeled array (a column): a NumPy array plus an index. A DataFrame is a two-dimensional labeled table; each column is a Series and the rows share an index. The index (row labels) is what distinguishes Pandas from a raw NumPy array.
</details>

### Q20. [Core] `.loc` versus `.iloc`?

<details><summary>Answer</summary>

`.loc` selects by label (index and column names): `df.loc[5, "age"]` uses the row whose label is 5. `.iloc` selects by integer position: `df.iloc[0, 2]` is the first row, third column, regardless of labels. They differ whenever the index is not a clean 0-based range (for example after filtering, or a date index). `.loc` is also inclusive of its stop label, while `.iloc` is exclusive like normal Python slicing.
</details>

### Q21. [Deep] What is the Pandas copy trap (chained assignment), and how do you avoid it?

<details><summary>Answer</summary>

Chained indexing like `df[df.a > 0]["b"] = 1` operates on a temporary intermediate that may be a copy, so the write may silently not stick (the classic SettingWithCopyWarning). Do the selection and assignment in one indexer: `df.loc[df.a > 0, "b"] = 1`. And when you derive a subset you intend to modify independently, take an explicit `.copy()` so you are not accidentally writing back to (or warned about) a view of the original.
</details>

### Q22. [Core] How do you handle missing values in Pandas?

<details><summary>Answer</summary>

Missing data shows as `NaN`. Detect with `.isna()`, drop with `.dropna()` (rows or columns), or fill with `.fillna(value)` (a constant, a column mean, forward or backward fill). The choice is a modeling decision: dropping loses data, filling invents it, so you pick per column based on why it is missing. `read_csv(..., na_values=...)` also lets you declare which raw tokens count as missing on load.
</details>

### Q23. [Core] What does `groupby` do, mechanically?

<details><summary>Answer</summary>

Split-apply-combine: split the rows into groups by the key column(s), apply an aggregation to each group (`mean`, `sum`, `count`, or a custom function), and combine the results into a new indexed table. `df.groupby("country")["cases"].sum()` gives one total per country. It is the table equivalent of a SQL `GROUP BY`.
</details>

### Q24. [Applied] You need one summary number per category and then the top category. Which calls?

<details><summary>Answer</summary>

`totals = df.groupby("category")["value"].sum()` for the per-category number, then `totals.idxmax()` for the label of the largest and `totals.max()` for its value. `idxmax` returns the index label (the category), which is usually what you want to report, not just the number.
</details>

---

## Part 5: APIs and the web (class 1.5)

### Q25. [Core] Sketch a GET request with `requests` and how you read the response.

<details><summary>Answer</summary>

```python
import requests
r = requests.get(url)
if r.status_code == 200:
    data = r.json()      # parse JSON body into a dict
    value = data.get("field")   # .get avoids a KeyError if absent
```

Check `.status_code` (200 is success), parse `.json()` into a dict, and pull fields with `.get(key, default)` so a missing key does not crash. A POST with a JSON body is `requests.post(url, json=payload)`, which is the exact shape of an LLM API call.
</details>

### Q26. [Deep] How should you handle an API key, and why?

<details><summary>Answer</summary>

Read it from an environment variable (`os.getenv("GROQ_API_KEY")`), never hard-code it, and never commit it (keep it in a `.env` that is git-ignored). Hard-coded keys leak through version control and shared notebooks and are hard to rotate. The habit, keys in the environment, is what makes the same code safe to share and to move between local and hosted providers later in the course.
</details>

### Q27. [Applied] A JSON response sometimes omits a field and your code crashes with KeyError. Fix?

<details><summary>Answer</summary>

Use `data.get("field")` (returns `None`) or `data.get("field", default)` instead of `data["field"]`, which raises when the key is absent. For nested optional structure, get each level with a default, or guard with `in`. Assume external responses are not guaranteed to contain every field.
</details>

---

## Rapid-fire (mixed tiers)

<details><summary>Open</summary>

- **Tuple vs list in one line.** Tuple is immutable and hashable; list is mutable.
- **Why can a tuple be a dict key but not a list?** Keys must be hashable (immutable); lists are not.
- **`==` vs `is` for `None`?** Always `is None`.
- **What does `with` guarantee?** The resource (file) is closed even if an error is raised.
- **Trailing `if` vs front `if/else` in a comprehension?** Trailing filters; front chooses the value.
- **`axis=0` in a DataFrame aggregation?** Collapses rows, so it operates down each column.
- **View or copy: basic NumPy slice?** View. Boolean or fancy index? Copy.
- **`.loc` vs `.iloc`?** Label vs integer position.
- **One-line fix for SettingWithCopyWarning?** Use a single `df.loc[rows, col] = ...`.
- **Where do API keys live?** In an environment variable, never in code.
- **What type is a parsed JSON object?** A dict.
- **`json.dumps` vs `json.dump`?** `dumps` returns a string; `dump` writes to a file.

</details>

## Senior / stretch questions (beyond core coverage)

### S1. [Senior] What is a generator, and when would you prefer it over a list?

<details><summary>Answer</summary>

A generator produces values lazily, one at a time, holding only the current item in memory rather than the whole sequence (`(x*x for x in xs)`, or a function using `yield`). Prefer it for large or streaming data, pipelines, or when you may stop early, because it saves memory and can start producing before the whole input is ready. The trade-off: you can iterate it only once and cannot index it.
</details>

### S2. [Senior] Why is vectorized NumPy or Pandas usually far faster than an equivalent Python loop?

<details><summary>Answer</summary>

The work runs in precompiled C over contiguous, fixed-dtype memory, avoiding the per-element overhead of the Python interpreter (dynamic typing, object boxing, bytecode dispatch) and using cache-friendly memory access and often SIMD. A Python loop pays that overhead on every element. This is why "don't loop over rows" is the first Pandas performance rule; reach for vectorized column operations or `groupby` instead.
</details>

### S3. [Senior] `df.apply` versus a vectorized operation: which and why?

<details><summary>Answer</summary>

Prefer a vectorized column operation (`df["a"] + df["b"]`, `np.where`, built-in string or datetime accessors) whenever one exists: it runs in C and is fastest. `df.apply` with a Python function runs that function per row or column in the interpreter, so it is flexible but slow, a fallback for logic that has no vectorized form. Row-wise `apply` (`axis=1`) is the slowest and worth avoiding on large frames.
</details>
