# Class update draft — not sent

Team 4 / RQ4.1 update for October 29:

Our evaluation logging and paired-bootstrap foundation is implemented and available
for review in draft PR https://github.com/vorhersager/gw1b-f2026/pull/4.
The exploratory protocol is `rq41/PROTOCOL_PILOT.md`; checkpoint and dataset-revision
manifests are included in the branch. It uses standard zero-shot HellaSwag, ARC-Easy
and PIQA, identical examples across checkpoints, raw and character-normalized
accuracy, and 10,000-replicate individual, unadjusted paired bootstrap intervals.

We are using existing trained class checkpoints for a preliminary 50M, single-seed
noise table. The previously pending tiny training job was canceled without running.
Real-evaluation results will be reported only after restoration, count and
repeatability checks pass. This pilot does not complete the three-seed 100M milestone.
Insert the verified Slurm job status and reviewed aggregate table link before sending.

## Seed ownership/budget request — draft

Han / Team 3: are three matched 100M training seeds already available? Please provide
run/checkpoint paths, actual token counts, initialization/data-order seeds and the
common training horizon. If unavailable, please confirm who owns the training and
which team's GPU budget covers it. We should avoid duplicate training and should
not call the three-seed milestone complete until these runs and results exist.

## Question collection coordination — draft

Yusra: please confirm the class-question form link, whether collection is open, and
who owns review/storage. If there is no form yet, the proposed fields are: question,
answer choices (when applicable), gold answer, explanation/reference, subject,
authorship date and optional contributor identifier. Collect original, verifiable
questions with no private information; keep development and final pools separate.
The research target is 300–500 reviewed questions. Authorship after the corpus cutoff
does not by itself establish that the underlying content is uncontaminated.

## Milestone status wording

- Logging/statistics code and local tests: completed; draft PR open.
- Pegasus guide: supplied and reviewed.
- Preliminary real-model noise table: pending actual evaluation output; update honestly.
- Protocol announcement/question form: drafts prepared; distribution/open status unverified.
- Three-seed 100M study: owner, budget and checkpoint availability still need confirmation.
