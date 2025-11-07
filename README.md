# MCQ Generator for Seeing Culture Benchmark

## Seeing Culture: A Benchmark for Visual Reasoning and Grounding

![Seeing Culture: A Benchmark for Visual Reasoning and Grounding](https://seeingculture-benchmark.github.io/static/images/teaser-white.png "Teaser")

See the project website for more information.
https://seeingculture-benchmark.github.io/

## Multiple Choice Question Generator Setup

```bash
$ source setup.sh
```

## Usage

```bash
$ source generate.sh
```

## Output

1. The output will be saved in the `./` directory in `questions_{timestamp}.jsonl` format.
2. Please download `data.zip` and unpack to the same directory as the `generate.sh` file.
3. You can visualize using the `visualizer.ipynb` file (please change the `JSON` file name accordingly).

## Citation
```
@inproceedings{satar-etal-2025-seeing,
    title = "Seeing Culture: A Benchmark for Visual Reasoning and Grounding",
    author = "Satar, Burak  and
      Ma, Zhixin  and
      Irawan, Patrick Amadeus  and
      Mulyawan, Wilfried Ariel  and
      Jiang, Jing  and
      Lim, Ee-Peng  and
      Ngo, Chong-Wah",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1131/",
    pages = "22238--22254",
    ISBN = "979-8-89176-332-6",
    abstract = "Multimodal vision-language models (VLMs) have made substantial progress in various tasks that require a combined understanding of visual and textual content, particularly in cultural understanding tasks, with the emergence of new cultural datasets. However, these datasets frequently fall short of providing cultural reasoning while underrepresenting many cultures.In this paper, we introduce the Seeing Culture Benchmark (SCB), focusing on cultural reasoning with a novel approach that requires VLMs to reason on culturally rich images in two stages: i) selecting the correct visual option with multiple-choice visual question answering (VQA), and ii) segmenting the relevant cultural artifact as evidence of reasoning. Visual options in the first stage are systematically organized into three types: those originating from the same country, those from different countries, or a mixed group. Notably, all options are derived from a singular category for each type. Progression to the second stage occurs only after a correct visual option is chosen. The SCB benchmark comprises 1,065 images that capture 138 cultural artifacts across five categories from seven Southeast Asia countries, whose diverse cultures are often overlooked, accompanied by 3,178 questions, of which 1,093 are unique and meticulously curated by human annotators. Our evaluation of various VLMs reveals the complexities involved in cross-modal cultural reasoning and highlights the disparity between visual reasoning and spatial grounding in culturally nuanced scenarios. The SCB serves as a crucial benchmark for identifying these shortcomings, thereby guiding future developments in the field of cultural reasoning. https://github.com/buraksatar/SeeingCulture"
}
```
