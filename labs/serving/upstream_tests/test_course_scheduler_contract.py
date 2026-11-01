"""Course-authored tests for pinned vLLM; copy into its tests/v1/core directory.

Not part of the course's dependency-free unittest suite. Requires the pinned
vLLM development environment plus locally available model configuration.
No model forward or GPU benchmark is performed by these tests.
"""
import os
import pytest
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test
MODEL = os.environ.get('COURSE_VLLM_TEST_MODEL', 'facebook/opt-125m')


@pytest.fixture(autouse=True)
def course_optional_mutation(monkeypatch):
    # An explicit negative control: mutate only the test process, never source.
    if os.environ.get('COURSE_MUTATE_SCHEDULER') != '1':
        return
    original = Scheduler.schedule
    def wrong_total(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.total_num_scheduled_tokens += 1
        return result
    monkeypatch.setattr(Scheduler, 'schedule', wrong_total)


def empty_result(output):
    ids = list(output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid:i for i,rid in enumerate(ids)},
        sampled_token_ids=[[] for _ in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize('budget,expected', [(512,[400,112]),(1024,[400,400,224])])
def test_course_partial_prefill_obeys_both_budgets(budget, expected):
    scheduler=create_scheduler(model=MODEL,max_num_batched_tokens=budget,
                               long_prefill_token_threshold=400,
                               enable_prefix_caching=False,
                               max_model_len=2048)
    requests=create_requests(num_requests=3,num_tokens=800,ignore_eos=True)
    for request in requests:scheduler.add_request(request)
    output=scheduler.schedule()
    counts=list(output.num_scheduled_tokens.values())
    assert counts==expected
    assert sum(counts)==budget
    assert output.total_num_scheduled_tokens==sum(counts)
    assert all(0<count<=400 for count in counts)
    scheduler.update_from_output(output,empty_result(output))
    scheduler.finish_requests(None,RequestStatus.FINISHED_ABORTED)
    assert scheduler.get_num_unfinished_requests()==0


def test_course_abort_running_request_is_idempotent():
    scheduler=create_scheduler(model=MODEL,max_num_batched_tokens=64,
                               enable_prefix_caching=False,max_model_len=2048)
    pool=scheduler.kv_cache_manager.block_pool
    free_before=pool.get_num_free_blocks()
    (request,)=create_requests(num_requests=1,num_tokens=128,ignore_eos=True)
    scheduler.add_request(request)
    output=scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert pool.get_num_free_blocks()<free_before
    scheduler.update_from_output(output,empty_result(output))
    first=scheduler.finish_requests(request.request_id,RequestStatus.FINISHED_ABORTED)
    free_after=pool.get_num_free_blocks()
    second=scheduler.finish_requests(request.request_id,RequestStatus.FINISHED_ABORTED)
    assert len(first)==1 and second==[]
    assert free_after==free_before
    assert pool.get_num_free_blocks()==free_after
    assert request.request_id not in scheduler.requests
    assert request.request_id not in scheduler.schedule().num_scheduled_tokens
