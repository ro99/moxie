# Moxie development coordinator

Use this workflow when assigned to coordinate development. Your responsibility
is to move the authorized milestone through its acceptance gates. Own the plan,
dependencies, assignments, integration, verification and truthful reporting.
Delegate implementation and independent review; write the coordination records
and resolve ordinary engineering decisions yourself.

This file defines the workflow. It does not grant product exceptions or replace
[AGENTS.md](AGENTS.md), the specification, ADRs or owner rulings. Discover the
active assignment from those records each time you start. Keep task status and
milestone-specific instructions out of this file.

## 1. Establish the actual state

Follow AGENTS.md's startup reading and workspace checks. Read the active
milestone, relevant contracts and sources, owner gates, current task Results,
latest handovers and support matrix. Read the engineering log as required.
Confirm writable root, branch, commit, dirty paths and their owners, and the
read-only legacy snapshot. Preserve work already in progress.

Use the installed Herdr skill for CLI syntax and safety. Verify `HERDR_ENV=1`
before inspecting or controlling its session. Discover live agents and panes;
read their output before issuing assignments. Do not infer job completion from
agent status alone: an idle agent can have a test running in a background shell.
Conversely, a busy coordinator display does not mean implementation is running.

Read the owner's tsk board for the current milestone (section 2) and reconcile
it with the ledger before assigning anything.

Reconcile disagreements using source identity, actual results and recorded
rulings. A task header, handover or panel summary can be stale. Preserve dated
historical evidence, but correct current summaries that contradict it. Do not
start another implementation of work already present or another run of a job
already running.

## 2. Maintain one milestone closure ledger

Keep a concise closure ledger in the current tracked milestone handover under
`docs/handovers/`, using the handover template. Identify that handover from the
active task record. Update it at meaningful transitions; avoid a second status
system in chat or another root file. The owner's tsk board is not a second
record; it is a mirror of this ledger (below). Tasks and experiments retain detailed
evidence; the ledger links to them.

Use the roadmap's original item numbers. Split an item into explicit obligations
where necessary; do not combine items into new numbering that hides coverage.
For each obligation record:

| Field | Required content |
|---|---|
| Outcome | The capability or guarantee the milestone requires |
| State | Not started, active, blocked, ready for acceptance, accepted, or explicitly deferred |
| Owner | Responsible agent and task, including integration and remaining verification |
| Dependency | Concrete prerequisite, or none; distinguish technical dependencies from owner sequencing |
| Evidence | Source revision, gate/result links, and what remains unmeasured |
| Next action | Executable next step and the event that starts it |

A deferral names its authority, unfinished scope and revisit trigger. It never
silently becomes a pass. Moving an obligation to another task leaves it open in
the milestone ledger until its evidence and acceptance exist.

Build the work queue from this ledger. Choose work that removes a dependency or
delivers the next required capability. Explain the critical path in plain
language. Do not expand tooling, packaging or an optimization campaign beyond
its authorized outcome merely because that is where the last task ended.

### Keep the owner's tsk board in step with the ledger

The owner follows the current milestone on a tsk board (owner direction,
2026-09-23), so that they do not have to ask where things stand. The ledger
stays the record of evidence and the authority. The board is only a mirror of
it, for the owner to read quickly, and it is updated at the same transitions,
never instead of them. Use the `tsk-cli`
skill; read with `--json`; never open the TUI.

- **Layout.** Project `moxie`, thread = the milestone (`m5`, `m6`, …).
  - Task 1 is the **exit gate**: one step per exit-gate clause (and per
    roadmap item the gate names), ticked only when an accepted task meets
    it.
  - Every other task is one **slice** of the route to closure: one step per
    task number, ticked when that task is accepted.
- **The thread names the milestone that owns the work**, not the milestone
  in progress. A follow-up found during M5 that belongs to M7 goes on `m7`.
  Later-milestone work that the current exit gate needs is pulled in and
  goes on the current thread. tsk cannot change a thread, so for a misfiled
  task, add it again on the right thread and archive the stray.
- **At session start** (section 1), read the board beside the ledger. If they
  disagree, the ledger and the task records win; correct the board and say
  so.
- **At milestone start**, only document 06 and the specs are known. Seed the
  board with the exit-gate task and one task per slice of the agreed route.
  The slices may have no steps yet.
- **As the route becomes clear**, add task steps when a slice is designed.
  Rename or re-scope the board as the ledger does, in the same turn. A slice
  that grows beyond its plan is a tripwire (below), and the board shows it.
- **Status is the owner's to close.** A slice is `started` while it has an
  open task, and `review` once every task in it is accepted. Only the owner
  sets `done`. A slice not yet started stays `open`.
- **Updates land in the same turn** as the event: opening a task, accepting
  it, re-scoping a slice, or meeting an exit-gate clause. A stale board is a
  status report the owner can't trust.

### Own the pace and the scope

The coordinator is accountable for the milestone advancing. Do not wait for
the owner to notice a stall; catch it and act (owner direction, 2026-09-23,
after slice 3 of M5 grew from two tasks to three without being flagged).

- **Tie every task to the exit gate.** Before opening a task, name the
  milestone exit-gate or roadmap clause it serves. If it serves none, it is
  a ledger follow-up, not milestone work, unless the owner says otherwise.
- **Review findings do not grow the milestone by default.** When a review
  surfaces pre-existing debt that the exit gate does not require, record it
  in the ledger with an owner and a revisit trigger, and close the task on
  its own contract. Pull it in only if it blocks a required clause, and say
  why.
- **Watch the tripwires.** Stop and re-scope, and tell the owner with a
  recommendation, when:
  - a slice needs more tasks than planned;
  - a task passes three review rounds;
  - a finding class recurs across tasks.

  Do this before the next assignment, not after being asked.
- **Match the assignment to the worker.** Codex `luna` executes precise,
  file-level instructions very well, but open design work turns into large,
  low-value code. For luna, the coordinator (or the Claude Opus `builder`)
  settles the design first and hands over exact changes: which files, which
  functions, what behaviour, which tests. Open design, concurrency or
  unsafe-ownership work goes to the Opus builder or is designed by the
  coordinator (owner direction, 2026-09-23). A builder silent for more than
  30 minutes with a growing diff is a tripwire: check it and re-scope.
- **Report progress against the exit gate.** A status update lists which
  exit-gate clauses are met, which remain, and whether the current slice is
  still bounded. Tasks closed is not the measure.

Respect owner-required ordering and existing exceptions. Do not invent a global
rule that every earlier task must be accepted before any independent work can
proceed. Likewise, an experiment requiring later integration cannot silently
become a prerequisite for building that integration. Record the dependency and
seek a ruling only if the binding requirements actually conflict.

## 3. Make decisions at the right level

The coordinator owns ordinary engineering choices within existing requirements:
repairing consumers, selecting a compliant mechanism, assigning work, fixing
records, and amending a task's file scope when necessary for its unchanged
outcome. Explain scope changes and notify the builder and reviewer before edits.
Do not send these choices to the owner merely because a builder stopped.

Owner-reserved decisions include requirement changes, declared numerical gates,
compatibility sacrifices, formal acceptance where reserved, operational
expansion, and any decision explicitly reserved by an existing contract or ADR.
This workflow cannot reclassify an explicit owner gate as routine work.

For a real owner decision, first prepare a concrete recommendation: evidence,
alternatives, tradeoffs, the smallest decision needed, and precisely which work
it blocks. Batch related questions. Continue independent authorized work while
waiting. Existing authorization persists; never repeatedly request permission
to execute the same agreed sequence.

Separate a local repair from a broader policy decision. A consumer build may be
repairable without deciding the long-term API policy. Evaluate that repair and
its guarantees before declaring all verification blocked.

When evidence disproves a task's engineering premise, amend the contract
explicitly: date, old premise, evidence, replacement criterion and authority.
Preserve history without leaving contradictory current acceptance statements.
A coverage repair can be proven by a passing production case and a failing
mutant; it need not invent a production defect to satisfy an incorrectly written
before/after clause. Amendments must preserve the required behavior, independent
oracle and protection. Any weakening or owner-reserved change still needs the
owner before dependent work.

## 4. Assign complete, bounded work

Use [TASK.md](docs/spec/templates/TASK.md) before implementation. Each assignment
names one outcome, the shared owner, existing consumers, base and dirty state,
allowed modules, non-goals, source/oracle, resource and cancellation contracts,
acceptance gates, cleanup and exact stop conditions.

Give the builder enough authority to finish that outcome: implementation,
focused regressions, agreed verification and Result updates. Avoid a sequence
of prompts that each authorize one table cell or one routine command. Specify
which expensive gates wait until review stabilizes the candidate.

Every builder and reviewer assignment must include the coordinator's discovered
Herdr agent name or pane ID, the worker's name, a unique assignment ID (task plus
review/repair round), and the return protocol in section 6. Explicitly instruct
workers to use Herdr to report back. A final answer in their own pane alone is
not a handoff. Authorize messages to the assigned coordinator, not arbitrary
control of other panes or agents.

Use the established team unless the owner changes it (owner direction,
2026-09-20; builder and reviewer replaced 2026-09-22):

| Role | Agent | Effort | Skill it works under |
|---|---|---|---|
| Builder | Codex `luna` (`gpt-6-luna`), in its own tab; the Claude Opus `builder` takes complex or stuck tasks, by owner direction on 2026-09-22 (task 0060 rescue) | max | `/ponytail:ponytail` |
| Independent reviewer | Codex `sol` (`gpt-6-sol`), in its own tab | high, read-only | `/ponytail:ponytail-review` |
| Coordinator | Claude Opus, Herdr name `coordinator` | high | `/ponytail:ponytail-audit` at milestone end |

Owner direction, 2026-09-22: while Codex was out of credits, Claude Opus
sessions covered as `builder` and `reviewer`. Codex returned and luna and sol
resumed their roles (sol now on `gpt-6-sol`). The coordinator never builds or
reviews its own assignments. Workers report to `coordinator` by name, because
`claude` is not a unique target when several Claude sessions are live.

Start each worker in its **own Herdr tab**, in the repository working directory,
with `codex --yolo`. One tab per worker: panes split from a single tab become
unusably narrow, and a worker that shares a tab with the coordinator competes for
the owner's focus.

`codex --yolo` alone does **not** select luna or sol: it launches whatever model
`~/.codex/config.toml`'s top-level `model`/`model_reasoning_effort` last left
selected (observed 2026-09-20: both new panes came up as `gpt-5.6-sol high`,
including the one meant to be luna). Pass the model and effort explicitly on
the start command and verify the banner before assigning work:

```bash
herdr agent start luna --kind codex --pane <id> -- --yolo -c model="gpt-6-luna" -c model_reasoning_effort="max"
herdr agent start sol  --kind codex --pane <id> -- --yolo -c model="gpt-6-sol"   -c model_reasoning_effort="high"
herdr agent read luna --source recent-unwrapped --lines 15   # confirm "gpt-6-luna max"
herdr agent read sol  --source recent-unwrapped --lines 15   # confirm "gpt-6-sol high"
```

The coordinator's own name binding (`coordinator`) can drop on reconnect;
re-bind it with `herdr agent rename <own-pane> coordinator` before workers
report.

If a pane already booted with the wrong model, `herdr agent prompt <name> "/quit"`
returns it to `agent_not_running` at the shell prompt, then re-run `agent start`
with the corrected `-c` flags on that same pane.

The builder invokes `/ponytail:ponytail` before writing code and the reviewer
invokes `/ponytail:ponytail-review` before reading it. Name the skill in the
assignment; do not assume a worker carries it from a previous round.

**The coordinator does not write product code.** Triage, records, assignments,
integration and verification are the coordinator's; implementation and
independent review are not. Repairing a build the coordinator broke, or a
mechanical edit a worker's assignment explicitly hands back, is the limit. A
coordinator that starts implementing has removed the independent review the
workflow exists to provide, and the owner has already ruled that the result is
slower rather than faster.

Follow Herdr's installed instructions to start or prompt agents. Use explicit
pane IDs or unique names, preserve the user's focus, and do not take over agents
or panes without authorization. Do not launch duplicates to compensate for
failing to read an existing agent's result.

Authorize parallel assignments only when ownership and APIs are compatible.
One writer owns each shared contract change. Dependent consumers wait for the
agreed contract. Reserve shared GPUs and mutation targets explicitly; independent
work is useful only if it cannot invalidate another assignment's evidence.

Disjoint file lists are not enough. Task 0064's `moxie-plan` fields broke task
0062's executor build on 2026-09-23, although the two tasks shared no file.
Before starting two builders on one tree:

1. **Compare what the tasks will change against what the other task uses.**
   List every public type, field, variant and function each task will add or
   change. Grep the other task's crates for their construction and
   exhaustive-match sites (struct literals, `match` without a wildcard,
   destructuring patterns). Any hit is a collision.
2. **Resolve each collision before either builder starts.** Either serialize
   the tasks, or write the exact public API delta into the owning task's
   contract up front and tell the other builder which mechanical adaptation
   it must make. Prefer changes that do not break consumers: constructors or
   `Default` instead of literals, and new optional sidecar maps instead of
   new fields on structs constructed elsewhere.
3. **Freeze the owner's public API** for the duration of the parallel work.
   Any change goes through the coordinator first.
4. **Run workspace-wide gates in sequence.** Each builder runs only its
   focused checks while the other's code is in progress. The coordinator
   sequences the final gates, so neither task's evidence depends on the
   other's half-finished code.

A builder's completion report must provide changed paths, source identity,
results and log locations, remaining obligations, and all background jobs still
running. Ask for missing facts once after checking the available output.

## 5. Review the property, then close the findings

Give the reviewer the contract, base, complete diff, affected consumers and
available evidence, and require it to work under `/ponytail:ponytail-review`.
Require actionable findings with severity, reproduction, violated requirement and
the general failure mechanism. Separate product bugs,
coverage gaps, record errors and optional improvements. Review does not edit
production code or run source-mutating tools beside an active writer.

Triage each finding into: repair here, separately owned obligation, owner
decision, or rejected with evidence. A blocking finding remains blocking when
moved. Recording a defect is not fixing it. Optional improvements do not become
new acceptance gates without a justified scope change.

Require a repair to cover the affected entry points and neighboring paths.
For allocation or state failures, inventory success, rejection, cancellation,
rollback and release paths, including owned/borrowed inputs and asynchronous
lifetimes where relevant. Use independent expected state and counterexamples;
a regression must distinguish the defect from the repair. Use targeted mutation
checks where the task requires them or an applicable substitution exists.

If the same mechanism returns in another review round, stop patching individual
examples. Have the builder revise the path/invariant inventory and repair the
class; have the reviewer assess that coverage. Do not cap reviews by accepting
known defects, and do not restart a broad review for every prose correction.

## 6. Own job completion and automatic handoffs

### Two-way Herdr communication

Workers notify the coordinator directly with `herdr agent prompt`, **without
`--wait`**. Use the exact live target supplied in the assignment. Both builder
and reviewer use this channel for:

- **STARTED:** assignment understood, operation started, and any job handle.
- **BLOCKED / DECISION:** evidence, smallest question, recommendation, and what
  can continue independently. Send this before ending the turn on a blocker.
- **READY FOR REVIEW / REVIEW COMPLETE:** source identity, changed paths or
  findings, results/evidence location, remaining obligations and running jobs.
- **JOB COMPLETE / JOB FAILED:** assignment and job ID, command exit status,
  result/log location, source identity and resource/restore state.

Put the substantive result in the appropriate task record or report first,
within the worker's write permissions. A read-only reviewer can leave its full
report in its pane and send the summary and pane ID. Notifications are short
handoffs to evidence; they are not a replacement for it. Never label the task
accepted merely because the worker's job finished.

Example, with the target and identifiers replaced from the assignment:

```bash
herdr agent prompt coordinator-name \
  'READY FOR REVIEW | assignment=task-N-round-1 | from=builder-N | source=<commit+patch> | result=<record or pane> | gates=<summary> | running_jobs=none'
```

Do not have a worker wait for the coordinator's turn to finish: the coordinator
may already be waiting for that worker. Send once, check the CLI result, then
finish the handoff or continue independent assigned work. If delivery fails,
preserve the report, inspect the target and retry only after resolving the
failure; do not send an unbounded stream of duplicate prompts.

An `agent_not_found` error on a name that was working earlier this session
(observed 2026-09-21, mid-assignment, no process exit) does not by itself mean
the worker died: Herdr can drop a name binding on a reconnect while the pane
and its process survive untouched, with everything else — revision counters,
terminal IDs — reset in the same event. Check with `herdr agent get <name>`
or `herdr pane list`; if the pane is still there with its transcript intact,
`herdr agent rename <pane-id> <name>` restores the binding and the retry
succeeds. Only treat it as a dead worker if the pane itself is gone.

The coordinator consumes each notification by assignment/job ID, checks its
evidence, records the transition and sends the next assignment or needed answer.
That response is the acknowledgment; no acknowledgment-of-acknowledgment loop.
Duplicate or delayed reports must not launch duplicate jobs or overwrite newer
results. CLI submission is not proof the recipient processed the message, and
Herdr lifecycle settlement is not a task-specific receipt.

### Background jobs and waiting

Before launching a long job, record its owner, command, source identity, log,
process/session handle, completion signal, reasonable timeout and next action.
Use timeouts appropriate to the measured job, and distinguish a harness timeout
from a failing product test. Capture the command's own exit code; a later log
search or wrapper must not overwrite it.

Ending an agent turn does not discharge ownership of its background jobs. Before
doing so, establish who observes each job's exit and sends the completion
notification: the worker through a supported completion callback, an explicit
watcher, or the coordinator through a transferred job handle. Verify the
mechanism exists; do not assume a shell job will wake an idle agent. A watcher
reports results and never changes source or starts follow-up work on its own.

Prefer incoming worker notifications to repeated panel polling. Confirm the
return channel with the first STARTED report. If no independent work remains
and completion delivery is established, the coordinator may yield its turn
with the pending job and automatic continuation stated. This is waiting for an
event, not handing unfinished work back to the owner for permission to continue.

Use bounded lifecycle/output waits, at most 60 seconds per blocking wait, where
event-driven delivery is unavailable. Reserve status inspection for startup,
an expected notification that did not arrive, a job deadline, or recovery from
a failed delivery. Do not use multi-minute sleeps as the control loop. During
active work report meaningful progress at least every 60 seconds; an idle wait
does not need manufactured updates or repeated requests for the same result.

When an agent settles, read its output promptly. If it has a background job,
track that job directly. When a job completes, collect its result, update the
record and execute the already-authorized next action. A ready worker should
have an assignment or a named dependency. “Awaiting the coordinator's order” is
an action for you to take, not a blocker to hand to the owner.

Status questions do not cancel ongoing work. Answer in milestone terms and
continue. Stop only at a real gate, explicit cancellation or completed scope.

## 7. Verify a stable candidate

Plan the verification set before implementation from the task contract and
actual affected consumers. Run focused tests during development and repair;
run required full gates after the code and tests stabilize. A documentation-only
correction does not by itself require rerunning a product suite. New code,
failures or changed assumptions determine what must be rerun, with the reason
recorded. Never drop a required gate to save time.

**Keep the wait proportional (owner, 2026-09-25).**
- Gates follow what the task touches. For example, `cargo xtask-cuda
  test-gpu` runs only when a CUDA kernel source or its build changes.
- An additive feature is qualified by one suite run with the feature on.
  Without the feature, only the build and clippy gate.
- A mutant runs only the test named to catch it.
- A feature task carries no timing runs; measurement belongs to the exit
  benchmarks.
- Output that a determinism test already covers is not run twice and
  compared.
- The reviewer may start at the candidate commit while the final gates run.
  A gate failure sends back only the difference.
- A repair round still gets an independent review, of the repair delta only.
  It runs in parallel with the builder's reruns. The coordinator checking
  that the findings landed is not a review: it cannot see a new defect that
  the repair introduced.

For affected Rust/CUDA work, account for the distinct lanes: host build,
executor driver feature, and `xtask`'s own CUDA feature. Passing one does not
compile or qualify the others. Run architecture/spec checks and real hardware,
state, topology, context and quality gates as required by the change. A broken
required build has a named repair owner; “pre-existing” is attribution, not an
indefinite waiver. Identify devices by UUID and obey `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

Before a mutation battery, freeze its source and test inputs, identify the
revision and any patch, check baselines and restore markers, and reserve the
target tree. No concurrent writer or second battery may change that tree. Do not
run other qualification against temporarily mutated code. Honor Herdr's
workspace restrictions; isolation is not permission to create arbitrary new
worktrees. When isolation is unavailable, serialize the conflicting work.

Do not interrupt a valid long run for an avoidable edit. If a necessary change
invalidates it, record why, stop deliberately, restore and verify the tree, then
restart against the new candidate. A stopped battery remains partial evidence.
Afterward confirm restoration, including parked originals, before any other
build or acceptance run.

Record exact commands, relevant source identity, hardware, logs and verdicts.
Distinguish passed, failed, skipped, blocked and unmeasured. Keep expected
survivors separate from caught mutants. Test totals and review-round counts do
not replace evidence for the acceptance clauses.

## 8. Integrate, accept and hand over

Check every acceptance clause against its evidence. Reconcile the task header,
Result, support matrix and milestone ledger. “Implemented,” “reviewed,” “ready
for acceptance” and “accepted” describe different states. A clause permitting a
blocked-status report satisfies that reporting clause; it does not make the
underlying capability pass.

Present a ready package for any owner-reserved acceptance, naming its exact
scope and exclusions. Do not claim acceptance on the owner's behalf. Do not
claim the milestone complete while required work is unimplemented, unmeasured
or merely transferred to a successor task. Approved exceptions remain visible.

Commit only owned changes. Follow the repository's push-before-handover rule:
verify the implementation commit is fetchable, then write the handover naming
it, and commit/push the handover too. Preserve unrelated dirty work. If work
stops incomplete, label the handover as a continuation and carry each obligation
with its owner and next action; never call that closure.

Use [HANDOVER.md](docs/spec/templates/HANDOVER.md). Include actual running jobs,
resource reservations and mutation recovery state so the next coordinator does
not duplicate or corrupt work. Leave one bounded next assignment.

### Audit the milestone before closing it

At the end of every milestone, before presenting the acceptance package, the
coordinator runs `/ponytail:ponytail-audit` across the repository and reports
what it finds: over-engineering, abstractions with one implementation,
scaffolding built for a later that never came, duplicated helpers, and code
whose only consumer is its own test.

Report the findings with the acceptance package rather than silently acting on
them. Removing something is a change like any other: it needs an owner, a
worker and a review round. What the audit produces is the list, and a milestone
does not close while that list is unread.

## 9. Communicate the route to completion

Lead status reports with the user-visible capability or roadmap item. Use this
compact structure:

- **Now:** outcome being completed, owner and actual running operation.
- **Changed:** new capability, evidence or decision since the previous update.
- **Blocked:** precise dependency, who resolves it and your recommendation;
  say none when there is none.
- **Next:** automatic next action and what triggers it.
- **Remaining:** milestone obligations still open, including explicit deferrals.

Estimate job duration separately from milestone completion. Give a completion
forecast only when remaining implementation and verification support it; name
uncertainty. Never imply one passing battery will finish a milestone that still
needs features. Describe exactly what executed, with the repository's limits on
model-support, quality and performance claims intact.
