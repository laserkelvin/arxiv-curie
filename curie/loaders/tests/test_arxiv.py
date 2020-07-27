from curie.loaders import arxiv


def test_arxiv_download():
    N_PAPERS = 5
    reader = arxiv.ArxivReader(keep=False)
    reader.query("astrochemistry", max_results=N_PAPERS)

    paper_count = 0
    for paper in reader.retrieve():
        paper_count += 1
    assert paper_count == N_PAPERS
