# EcoReason Portal

Ask an invasion ecology question, trace the answer to study passages, and check whether the evidence supports a causal estimate. The workbench searches the Zenodo corpus with DeepResearch and Qdrant. It uses EcoAIRE records for hypothesis turns and stores ecologist reviews as versioned decisions.

## Run with Docker

With Docker and Docker Compose installed, run this from the repository root.

```sh
docker compose up --build -d
```

Compose starts Streamlit, the API, and Qdrant together. On the first run, it imports the bundled Zenodo corpus of 12,636 papers, downloads the FastEmbed model files, and builds the Qdrant index. This can take several minutes or longer depending on your CPU. To follow progress, run `docker compose logs -f index`.

Open the chat when startup finishes. Later starts with `docker compose up -d` reuse the saved corpus, Qdrant index, chats, and reviews. [Open EcoReason Portal](http://127.0.0.1:7860)

## Use the portal

Select **Try a causal analysis** to compare two randomly assigned diets in a lab experiment and see the estimated effect, uncertainty, and supporting data. **Example questions** also includes the lake-restoration and enemy-release cases. Open **Review study claims** to accept, revise, or reject a study interpretation. The workspace URL restores up to ten recent turns and their saved review decisions.

Select **New chat** to start an empty workspace. The previous chat and its reviews remain available through their original workspace URL.

In offline mode, ordinary chat returns candidate passages from the corpus. With live models enabled, it shows a generated answer followed by any cited passages. If the corpus lacks relevant sources, the app labels uncited output as a general model answer. Turn on **Analyze a hypothesis** for structured evidence assessment and EcoAIRE records.

## What the app can estimate

The bundled analysis uses the 20 juvenile red swamp crayfish in the two complete cold-treatment arms of Carreira et al. (2017). It estimates plant-only minus animal-only diet for daily body mass change over two months at 17°C. The statsmodels regression adjusts for the sex balance in the design and reports an approximate 95% interval. The CSV in `examples/` is transcribed from the S1 data supplement. Other settings and outcomes require their own study comparisons. [Study design](https://doi.org/10.1371/journal.pone.0183108) · [S1 data table](https://doi.org/10.1371/journal.pone.0183108.s001)

The answer distinguishes a calculated estimate, a study-reported finding, and insufficient evidence for estimation. The prototype calculates a numeric effect for the bundled contrast. For free-form questions, offline mode retrieves passages. Live mode can also structure hypotheses, extract passage-backed comparisons, and assess their applicability.

## EcoAIRE in the workflow

The app uses the bundled EcoAIRE template's section names and validates the fields it relies on with local Pydantic models. Python runs the statistical analysis.

After intake, a hypothesis record provides the proposed intervention, comparator and outcome for DeepResearch retrieval and the evidence agents. The causal demo uses a registered lab experiment and runs its estimator only for a matching EcoAIRE study design, context and source. The workflow writes the estimate and interval to `evidence.effect_size` and links the evidence to the experiment and paper. A broader hypothesis can carry a quoted finding from the retrieved studies. Its missing fields stay unset.

Open **EcoAIRE record** under an answer to view or download the native `hypothesis_benchmark_instance` JSON. The analysis details and dataset DOI appear with the causal result in the chat.

## OpenRouter

**Offline is the default.** The forked DeepResearch loop uses local search and deterministic query generation in offline mode. To enable live models, set `ECO_ENABLE_LIVE=1` and `OPENROUTER_API_KEY` in a `.env` file next to `compose.yaml`, then run `docker compose up -d`. Live calls send the query and selected study text to OpenRouter and can incur charges.

| Role | Model | Override |
| --- | --- | --- |
| Intake | [`nvidia/nemotron-3.5-super-vl:free`](https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free) | `ECO_INTAKE_MODEL` |
| DeepResearch | [`nvidia/nemotron-3.5-super-vl:free`](https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free) | `ECO_DISCOVERY_MODEL` |
| Extraction | [`nvidia/nemotron-3.5-super-vl:free`](https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free) | `ECO_EXTRACTION_MODEL` |
| Assessment | [`nvidia/nemotron-3.5-super-vl:free`](https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free) | `ECO_ASSESSMENT_MODEL` |

Set any model overrides in the same `.env` file.

Live literature overviews use local retrieval and one synthesis call after intake. The synthesis receives short excerpts from up to six papers and returns findings with source IDs. The app checks those IDs and links each accepted finding to the original passage. If the model output is unusable, the chat opens the candidate passages.

All free defaults use Nemotron, including plain chat. The app sets its reasoning effort to `none` to reserve output tokens for JSON and answer text. When a role uses a `:free` model, hypothesis turns run one DeepResearch search and batch extraction for up to three papers, normally using at most four model requests. If a separate free intake model is configured and its provider throttles the request, intake can try the configured free extraction model once. If intake is rate-limited, the app can still run a local search. A rate-limited literature overview keeps its retrieved passages available. Type a new question about the public corpus to exercise the live agents. The example buttons run locally. [OpenRouter limits](https://openrouter.ai/docs/api_reference/limits) · [NVIDIA free endpoint terms](https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free)

## Architecture and checks

Streamlit calls FastAPI. LangGraph structures the request, runs DeepResearch against the local index, extracts quoted comparisons, and assesses the evidence. Qdrant combines FastEmbed dense and BM25 rankings and returns the best matching passage per paper. The causal demo computes its effect from the bundled study observations in Python. SQLite stores the source corpus, LangGraph checkpoints, and versioned reviews. 

```sh
ECO_ENABLE_LIVE=0 uv run --locked pytest -q
uv run --locked ruff check ecoportal tests
```

## Limitations and next Todos

- **Models choice.** Models of free tiers are currently used from openrouter for the application. Further analysis of the results from the models in each stage and further research on more qualified models is needed for each role as well looking into fine tuning them the models especially for the asseessment model.
- **Causal scope.** The portal estimates one randomized diet contrast for juvenile crayfish at 17°C. Live mode assesses other comparisons using published passages. Add reproducible contrasts and document the assumptions needed for observational estimates and for applying results in new settings.
- **Retrieval coverage.** Dense search uses titles and the first 600 characters of abstracts. BM25 searches full abstracts and the available full text (2,834 of 12,636 papers). Benchmark embeddings of complete abstracts within the model’s token limit and of methods and results passages against ecologist-labelled evidence queries.
- **DeepResearch.** In free-model hypothesis turns, DeepResearch generates one query from the intake brief and the extraction agent reads passages from up to three papers. Compare this with recursive discovery and check whether retrieved quotes support the study designs and findings attributed to them.
- **Extraction tags.** The current DOI join gives 10,350 papers searchable species, location, habitat, ecosystem, or relationship labels. The importer restores only the first slash in archive filenames, leaving some tags unjoined. Fix the join for DOIs with multiple slashes and rebuild the index.

## Acknowledgments and sources

Thanks to Jennifer D’Souza at the TIB Leibniz Information Centre for Science and Technology and to the SciKnowOrg team and their research collaborators for the software, schema, and data this portal builds on.

- **DeepResearch Eco.** Jennifer D’Souza, Endres Keno Sander, and Andrei Aioanei (2025), *DeepResearch Eco: A Recursive Agentic Workflow for Complex Scientific Question Answering in Ecology*. The portal adapts the Python research loop to use local search and the portal’s model configuration. [SciKnowOrg repository](https://github.com/sciknoworg/deep-research) · [Paper](https://arxiv.org/abs/2507.10522) · 
- **EcoAIRE.** The bundled hypothesis benchmark template for the portal’s hypothesis, experiment, and evidence records comes from SciKnowOrg. [EcoAIRE template](https://github.com/sciknoworg/ecoaire)
- **Invasion biology corpus.** Jennifer D’Souza (TIB, 2024), *Invasion Biology WikiProject Scientific Papers: Text Data Mining and LLM-based Information Extraction of Species, Locations, Habitats, and Ecosystems*. The portal bundles `publications_tdm.csv` (compressed as `.xz`) and `structured-publications.zip` from Zenodo record and joins the records by DOI. Zenodo lists the release under Creative Commons Attribution 4.0 International. D’Souza compiled the corpus using ORKG Ask and the Invasion Biology WikiProject corpus of Mietchen et al. (2024). The accompanying `invasion-biology-IE` repository contains the upstream retrieval and extraction scripts. [Zenodo release](https://doi.org/10.5281/zenodo.14525619) · [Mietchen et al. dataset](https://doi.org/10.5281/zenodo.12518037) · [ORKG Ask](https://ask.orkg.org/) · [Invasion-biology-IE repository](https://github.com/jd-coderepos/invasion-biology-IE) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Corpus paper.** Jennifer D’Souza, Zachary Laubach, Tarek Al Mustafa, Sina Zarrieß, Robert Frühstückl, and Phyllis Illari (2025), *Mining for Species, Locations, Habitats, and Ecosystems from Scientific Papers in Invasion Biology: A Large-Scale Exploratory Study with Large Language Models*. NLP4Ecology 2025, pp. 16–23, © 2025 University of Tartu Library. [ACL Anthology](https://aclanthology.org/2025.nlp4ecology-1.6/) · [University of Tartu Library](https://hdl.handle.net/10062/107178)
- **Worked examples.** The crayfish diet data comes from Carreira et al. (2017). The lake-restoration and enemy-release passages come from van der Wal et al. (2013) and Roy et al. (2011). [Crayfish data supplement](https://doi.org/10.1371/journal.pone.0183108.s001) · [Lake restoration study](https://doi.org/10.1371/journal.pone.0078579) · [Enemy release review](https://doi.org/10.1007/s10526-011-9349-7)
