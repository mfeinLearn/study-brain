# study-brain

An MCP server that learns what you're unclear on from the questions you ask.

While studying (Flatiron AI engineering, DeepLearning.AI's Math for ML, LeetCode), you
ask an assistant a lot of questions. Many of them are the same confusion asked
in different words, weeks apart. study-brain logs each question, tags it with a topic,
spots repeats, and ranks the topics you're most unclear on. It also answers from your
own notes (RAG) with citations, so you can review the right material.

Everything runs locally. No API keys.

## What it does

- **`log_question(text, source)`** stores a question, tags its topic, and tells you if you've asked something like it before.
- **`search_my_notes(query, k)`** retrieves the k most relevant chunks of your notes, each with a `path#chunkN` citation.
- **`my_unclear_topics(limit=3)`** returns your top unclear topics with their scores, the repeated confusions, past questions, and related note chunks.
- **`similar_questions(text)`** finds past questions that look like the same confusion.

## Architecture

```
            Claude Desktop / Claude Code            CLI (python -m study_brain)
                         │ MCP (stdio)                         │
                         ▼                                     ▼
                 server.py (FastMCP) ───────────────►  brain.py  StudyBrain
                                                          │
         ┌───────────────┬───────────────┬────────────────┼────────────────┐
         ▼               ▼               ▼                ▼                ▼
    topics.py       clustering.py    scoring.py       ingest.py       storage
    keyword map     cosine sim +     unclear score    chunk .md/.txt  SQLite: questions
    (topics.yaml)   union-find       per topic        .py/.ipynb      Chroma: "questions"
                                                                       Chroma: "notes"
```

- **SQLite** (`questions`: id, text, topic, source, timestamp) is the source of truth for questions.
- **Chroma** has two collections, both using cosine distance:
  - `questions`: one embedding per logged question, used for repeat detection and `similar_questions`.
  - `notes`: chunks of your notes with metadata `source`, `chunk`, `topic`, `date`.
- **Embeddings** come from Chroma's default local embedding function (all-MiniLM-L6-v2 on ONNX Runtime). It downloads about 80 MB to `~/.cache/chroma` the first time you use it and runs on CPU after that.
- **Topic tagging** is deterministic: `src/study_brain/topics.yaml` maps each topic to keywords and phrases. A question gets the topic with the most whole-word keyword hits. Ties go to the topic listed first, and a question with no hits is tagged `other`. Topics: linear_algebra, calculus, probability_stats, machine_learning, python_pandas, react, dynamic_programming, graphs.
- **Repeat detection** links two questions when their embeddings have cosine similarity ≥ 0.6 (`STUDY_BRAIN_SIM_THRESHOLD`). It uses single linkage (union-find), so a chain of close paraphrases ends up in one cluster. On the sample data the labelled paraphrases score 0.62–0.93 against each other, and the closest unrelated pair scores 0.50.

Data lives in `~/.study_brain/` by default. Set `STUDY_BRAIN_HOME` to put it somewhere else.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/mfeinLearn/study-brain.git   # or unzip study-brain.zip
cd study-brain
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Try the demo with the sample notes and the 27 **sample** questions in
`src/study_brain/sample_data.py`. The questions are made up for the demo.

```bash
python -m study_brain ingest ./notes      # chunk + embed the notes folder
python -m study_brain seed                # load SAMPLE questions (replaces existing questions)
python -m study_brain report              # top 3 unclear topics
python -m study_brain ask "What is an eigenvector, intuitively?"
```

Other commands: `similar "..."`, `search "..."`, `report --json`, `report --limit 5`, and `serve`, which runs the MCP server.

Use your own notes with `python -m study_brain ingest ~/path/to/notes`. It reads `.md`, `.txt`, `.py`, and `.ipynb` files. Re-ingesting a file replaces its old chunks.

To avoid mixing the demo with real data, run the demo with a separate data dir, for example `STUDY_BRAIN_HOME=/tmp/sb-demo`.

## Connect to Claude Desktop

Edit `claude_desktop_config.json`:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "study-brain": {
      "command": "/absolute/path/to/study-brain/.venv/bin/python",
      "args": ["-m", "study_brain", "serve"],
      "env": {
        "STUDY_BRAIN_HOME": "/Users/you/.study_brain"
      }
    }
  }
}
```

Use absolute paths. Claude Desktop doesn't start in your project folder or activate your venv. Restart Claude Desktop after you change the file. Run `python -m study_brain ingest ./notes` once first, so the embedding model is already downloaded and the server's first call is fast.

Here's a prompt that works well: *"When I ask a study question, call log_question first, then use search_my_notes to answer from my notes and cite the sources."*

## Connect to Claude Code

```bash
claude mcp add study-brain \
  --env STUDY_BRAIN_HOME="$HOME/.study_brain" \
  -- /absolute/path/to/study-brain/.venv/bin/python -m study_brain serve
```

Check it with `claude mcp list`, or run `/mcp` inside Claude Code.

## Scoring

Each topic *t* gets an unclear score:

```
score(t) = w_f · N_t  +  w_r · Σ_{q ∈ t} 0.5^(age_days(q) / H)  +  w_c · C_t
```

| Term | Meaning | Default |
|---|---|---|
| `N_t` | number of questions tagged *t* (frequency) | |
| `0.5^(age/H)` | recency weight of one question. It's 1.0 today and halves every *H* days | H = 14 days |
| `C_t` | repeat clusters in *t*: groups of 2+ questions that are the same confusion | |
| `w_f, w_r, w_c` | weights | 1.0, 2.0, 3.0 |

The reasoning behind each term:
- **Frequency** counts how much you've asked about a topic.
- **Recency** makes current confusion count more than something you asked about a month ago and have probably worked out since.
- **Repeat clusters** carry the most weight. Asking the same thing again is the clearest sign it didn't stick. A topic with two scattered questions scores lower than a topic where you asked one question twice.

A cluster counts toward the topic most of its questions are tagged with. `other` is left out of the ranking by default. The weights are in `src/study_brain/config.py` (`ScoreWeights`).

## Tests and eval

```bash
pytest                      # tagging, chunking, clustering, scoring order, MCP tool smoke test
python eval/run_eval.py     # checks flagged topics against the sample ground truth and prints a score
python scripts/check_mcp.py # starts the server over stdio, lists its tools, and calls each one
```

The eval seeds the sample data into a temporary directory and reports three numbers:
- tagging accuracy
- pair precision, recall, and F1 for repeat clusters against the labelled paraphrase groups
- precision@3 of the flagged top topics against `EXPECTED_TOP_TOPICS`

The overall score is the mean of the three. The eval exits non-zero when the overall score is below 0.9.

## Project layout

```
src/study_brain/
  brain.py        core service (used by the CLI and the MCP server)
  server.py       FastMCP server with 4 tools
  cli.py          ingest / ask / report / similar / search / seed / serve
  topics.py       keyword tagger; topics.yaml holds the keyword map
  clustering.py   cosine similarity + union-find repeat clusters
  scoring.py      unclear score
  ingest.py       file loading (.ipynb flattening) and chunking
  db.py           SQLite questions table
  sample_data.py  SAMPLE questions with labels (ground truth for the eval)
  seed.py         loads the sample questions
notes/            sample notes
tests/            pytest suite
eval/run_eval.py  ground-truth eval
scripts/check_mcp.py  stdio MCP client check
```

## Limitations

- Keyword tagging is easy to inspect, but it misses questions that don't use any of the keywords. Those go to `other`. Add keywords to `topics.yaml` as you go.
- The 0.6 similarity threshold was tuned on the sample data. Short or very generic questions can land on either side of it.
- Clustering is O(n²) over all questions. That's fine for thousands of questions, but it isn't built for millions.

## Roadmap

- **LeetCode coach integration:** log questions automatically from a LeetCode practice session, tag them by problem pattern (DP, graphs, two pointers), and suggest the next problem that targets your weakest pattern.
- **`quiz_me` tool:** generate a short quiz from your top unclear topic using your own note chunks. Mark a topic as improving when you answer correctly.
- **Sunday summary:** a weekly digest of what you asked, which confusions repeated, which ones stopped coming up, and what to review next week.
- Optional LLM tagging as a fallback for questions tagged `other`.
- Track when a confusion stops recurring, to show progress over time.

## License

MIT. See [LICENSE](LICENSE).
