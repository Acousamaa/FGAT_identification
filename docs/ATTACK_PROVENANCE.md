# ATT&CK provenance and source mapping

FGAT-Bench was constructed using **MITRE ATT&CK Enterprise v16.1**. The original collection workflow used official ATT&CK technique/sub-technique pages rather than parsing STIX objects directly.

The information-source mapping used in the study is:

| Stage | Exact source |
|---|---|
| Benchmark construction | General descriptive narrative on each Enterprise technique or sub-technique page |
| Technique-name refinement | Official Technique Name or Sub-technique Name |
| Operational-semantic enrichment | Manually selected definitional phrases from the page Description section |
| Tool/command enrichment | Tool identifiers and procedural command strings from Procedure Examples |
| Testing | Fixed held-out FGAT-Bench split and independent TRAM dataset |

`resources/source_map.csv` provides the same mapping in machine-readable form.
