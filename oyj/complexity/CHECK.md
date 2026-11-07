# Hand check of seeds + dev (220 lines)

Method: two independent reviewers labeled all 220 briefs blind (no TypeSafe label shown). The controller compared with TypeSafe and decided every disagreement by reading the brief.

- n checked: 220 (blind double label); agreement with TypeSafe: 164/220
- n disagreements read by the controller: 56
- n changed: 53 (kept TypeSafe on 3: p1-033 and p4-163 = new function plus test, scoped; p3-076 = design brief)
- pattern: TypeSafe puts multi-file renames and mechanical replacements at level 1 (scoped); the level 0 text names renames, so gold = 0.
- limit: the 164 agreed lines were not re-read by the controller; two labelers agreed on them.

| uid | TypeSafe | gold | reason |
|---|---|---|---|
| p1-006 | 1 | 0 | delete debug prints |
| p1-012 | 1 | 0 | field rename keep alias |
| p1-025 | 1 | 0 | fix import path |
| p1-078 | 1 | 0 | docstring example output edit |
| p1-080 | 1 | 0 | text replacement in templates |
| p1-083 | 1 | 0 | fix relative doc links |
| p1-093 | 1 | 0 | version pin bump |
| p1-098 | 1 | 0 | header line replace |
| p1-126 | 1 | 0 | remove dead flag branch |
| p1-128 | 1 | 0 | label string change |
| p1-164 | 1 | 0 | single help string edit |
| p2-002 | 1 | 0 | lint autofix type imports |
| p2-012 | 1 | 0 | class rename |
| p2-033 | 1 | 0 | story title strings |
| p2-034 | 1 | 0 | console.log to logger swap |
| p2-050 | 1 | 2 | hydration mismatch diagnosis and fix |
| p2-070 | 1 | 0 | given SQL migration file |
| p2-080 | 1 | 0 | one log line |
| p2-098 | 1 | 0 | describe string rename |
| p2-104 | 1 | 0 | mechanical lowercase of hex colors |
| p2-111 | 1 | 0 | remove rolled-out flag branches |
| p2-113 | 1 | 0 | dev dependency bumps |
| p2-136 | 1 | 0 | mechanical prop rename |
| p2-138 | 1 | 0 | reformat JSON fixtures |
| p2-159 | 1 | 0 | mechanical annotation conversion |
| p2-173 | 1 | 0 | action version bump |
| p3-008 | 1 | 0 | binary name rename |
| p3-016 | 1 | 0 | mechanical header line change |
| p3-017 | 1 | 0 | compose service rename |
| p3-026 | 1 | 0 | add one terraform output |
| p3-031 | 1 | 0 | variable rename |
| p3-034 | 1 | 0 | tiny PDB yaml manifest |
| p3-042 | 1 | 0 | runner label swap |
| p3-050 | 1 | 0 | python version metadata bump |
| p3-051 | 1 | 0 | label value rename |
| p3-057 | 1 | 0 | field rename across listed files |
| p3-107 | 1 | 0 | license header replacement |
| p3-121 | 1 | 0 | field rename with alias |
| p3-122 | 1 | 0 | doc link path replace |
| p3-140 | 1 | 0 | add compose service |
| p3-141 | 1 | 0 | test function renames |
| p3-143 | 1 | 0 | env var prefix rename |
| p3-172 | 1 | 0 | add labels to Dockerfiles |
| p3-173 | 1 | 0 | strip trailing whitespace |
| p3-175 | 1 | 0 | constant change |
| p4-016 | 1 | 0 | gradle task alias rename |
| p4-042 | 1 | 0 | field rename two callers |
| p4-116 | 1 | 0 | package rename, mechanical moves |
| p4-124 | 1 | 0 | helm values add |
| p4-136 | 1 | 2 | open capacity estimate, judgment reasoning |
| p4-138 | 1 | 0 | mechanical anchor fixes |
| p4-160 | 1 | 0 | add npm script |
| p4-167 | 1 | 0 | terraform variable rename |
