import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from user_profile_pipeline.personalized_dialogue.runner.experiment_budget import BudgetLedger, BudgetExceeded
from user_profile_pipeline.personalized_dialogue.runner.interventions import matched_derangement, valid_scores, valid_assignments, blind_temporal_input, validate_returned_model, scoreable_assignments


class BudgetTests(unittest.TestCase):
    def test_concurrent_reservations_do_not_exceed_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'budget.sqlite',3)
            def reserve(i):
                try:return ledger.reserve(str(i),'test',1)
                except BudgetExceeded:return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                ids=list(pool.map(reserve,range(12)))
            self.assertEqual(sum(i is not None for i in ids),3)
            self.assertEqual(ledger.snapshot()['committed_ca'],3)

    def test_ambiguous_request_remains_charged_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.sqlite';ledger=BudgetLedger(path,2)
            call=ledger.reserve('x','test',2);ledger.fail(call)
            resumed=BudgetLedger(path,2)
            with self.assertRaises(BudgetExceeded):resumed.reserve('y','test',.01)
            with self.assertRaises(ValueError):BudgetLedger(path,3)

    def test_settlement_releases_unused_reserve_but_counts_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'budget.sqlite',5)
            call=ledger.reserve('x','test',5)
            ledger.settle(call,{'prompt_tokens':1000,'completion_tokens':500},1,2)
            self.assertEqual(ledger.snapshot()['metered_ca'],2)
            ledger.reserve('y','test',3)
            with self.assertRaises(BudgetExceeded):ledger.reserve('z','test',.01)

    def test_missing_usage_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'budget.sqlite',5);call=ledger.reserve('x','test',3)
            with self.assertRaises(ValueError):ledger.settle(call,{},1,2)
            self.assertEqual(ledger.snapshot()['committed_ca'],3)


class InterventionValidationTests(unittest.TestCase):
    def test_temporal_omission_is_an_error_not_a_dropped_target(self):
        rows, missing=scoreable_assignments({'assignments':[{'id':'I1','bucket':'stable'}]}, {'I1':'stable','I2':'recent'})
        self.assertEqual(missing,['I2'])
        self.assertEqual(rows[-1],{'id':'I2','bucket':'invalid'})
        with self.assertRaises(ValueError):scoreable_assignments({'assignments':rows}, {'I1':'stable','I2':'recent'})

    def test_provider_model_substitution_is_rejected(self):
        config={'allowed_returned_models':{'gpt-4o-mini':['gpt-4o-mini-2024-07-18']}}
        validate_returned_model(config,'gpt-4o-mini','gpt-4o-mini-2024-07-18')
        with self.assertRaises(ValueError):validate_returned_model(config,'gpt-4o-mini','gpt-4.1-mini-2025-04-14')

    def test_temporal_prompt_order_does_not_reveal_export_bucket_order(self):
        first={'temporal_input':{'interests':[{'id':'I0','interest':'coffee','supporting_posts':['P0']},{'id':'I1','interest':'hiking','supporting_posts':['P1']}]},'temporal_gold':{'I0':'stable','I1':'recent'}}
        second={'temporal_input':{'interests':[{'id':'I0','interest':'hiking','supporting_posts':['P1']},{'id':'I1','interest':'coffee','supporting_posts':['P0']}]},'temporal_gold':{'I0':'recent','I1':'stable'}}
        self.assertEqual(blind_temporal_input(first),blind_temporal_input(second))

    def test_wrong_user_mapping_is_a_bijection_without_self_matches(self):
        profiles=[{'food':{'stable':['coffee']*i,'recent':[]}} for i in range(1,6)]
        mapping=matched_derangement(profiles)
        self.assertEqual(sorted(mapping),list(range(5)))
        self.assertTrue(all(i!=j for i,j in enumerate(mapping)))

    def test_no_recent_gold_is_not_scored_as_zero(self):
        value={'coverage':None,'concreteness':3,'fluency':4,'rationale':'No target labels'}
        self.assertTrue(valid_scores(value,False));self.assertFalse(valid_scores(value,True))
        value['coverage']=0;self.assertFalse(valid_scores(value,False))
        value['coverage']=True;self.assertFalse(valid_scores(value,True))

    def test_temporal_duplicate_or_missing_ids_are_rejected(self):
        targets={'I1':'stable','I2':'recent'}
        self.assertFalse(valid_assignments({'assignments':[{'id':'I1','bucket':'stable'}]*2},targets))
        self.assertTrue(valid_assignments({'assignments':[{'id':'I1','bucket':'stable'},{'id':'I2','bucket':'recent'}]},targets))


if __name__=='__main__':unittest.main()
