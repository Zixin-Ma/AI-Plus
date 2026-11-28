"""Regression checks for candidate feasibility and paired deployment decisions."""
import unittest
import numpy as np
import guard
import policy
from industrial import macro_f1


class Controls(unittest.TestCase):
    def test_single_legal_choice_names_parent(self):
        scores={"parent":dict(primary=1,worst=1,floor=0,allowed=True),
                "candidate":dict(primary=0,worst=0,floor=-1,allowed=False)}
        prompt=policy.final_request("A",{},scores)[0]["content"]
        self.assertIn("parent|retain validated model",prompt)
        self.assertNotIn("candidate",prompt)

    def test_masked_history_and_active_candidate(self):
        entry=dict(primary=1,worst=.8,floor=0)
        scores={"parent":dict(entry,allowed=True),"history":dict(entry,allowed=False),
                "candidate":dict(entry,allowed=True)}
        text=policy.final_request("A",{},scores)[0]["content"]
        self.assertIn("parent, candidate",text)
        self.assertNotIn("history",text)

    def test_f1_absent_classes_and_ties(self):
        labels=np.array([0,0,2,2,2])
        guess=np.array([0,1,2,1,2])
        self.assertAlmostEqual(macro_f1(labels,guess,6),(2/3+0+4/5)/3)

    def test_paired_tail_is_zero_for_the_incumbent(self):
        x=np.stack([np.linspace(0,10,60),np.linspace(0,20,60)],axis=1)
        pred=np.arange(60.)[:,None]
        panel=guard.PairedRisk("B",x,x+1,np.zeros(60),pred,"relative")
        self.assertAlmostEqual(panel.score(pred),0)
        self.assertLess(panel.score(pred+2),0)


if __name__=="__main__":
    unittest.main(verbosity=2)
