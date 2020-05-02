from unittest import TestCase

from qdtp import *


class ProofTestCase(TestCase):
    def test_proving_modus_ponens(self):
        ante = {Var("P"),
                Cond(Var("P"), Var("Q"))}
        cons = {Var("Q")}
        Sequent(ante, cons).prove()

    def test_proving_a_tautololgy(self):
        ante = set()
        cons = {Cond(Var("P"),
                     Cond(Var("Q"), Var("P")))}
        Sequent(ante, cons).prove()

    def test_proving_double_negation(self):
        ante = {Var("P")}
        cons = {Not(Not(Var("P")))}
        Sequent(ante, cons).prove()

    def test_proving_contraposition(self):
        ante = {Cond(Var("P"),
                     Var("Q"))}
        cons = {Cond(Not(Var("Q")),
                     Not(Var("P")))}
        Sequent(ante, cons).prove()

    def test_proving_transitivity_of_conditionals(self):
        ante = {Cond(Var("P"), Var("Q")),
                Cond(Var("Q"), Var("R"))}
        cons = {Cond(Var("P"), Var("R"))}
        Sequent(ante, cons).prove()

    def test_proving_de_morgans_laws(self):
        ante = {Not(Or(Var("P"),
                       Var("Q")))}
        cons = {And(Not(Var("P")),
                    Not(Var("Q")))}
        Sequent(ante, cons).prove()

        ante = {Not(And(Var("P"),
                        Var("Q")))}
        cons = {Or(Not(Var("P")),
                   Not(Var("Q")))}
        Sequent(ante, cons).prove()

    def test_finding_counter_examples(self):
        ante = {Var("P"),
                Cond(Var("Q"), Var("P"))}
        cons = {Var("Q")}

        """
        P   Q  |  P   Q -> P   Q
        -------+----------------
        T   T  |   T     T
        T   F  |   T     T      F *
        F   T  |   F
        F   F  |   F
        """

        try:
            Sequent(ante, cons).prove()
        except CounterExample as counter:
            self.assertTrue(counter["P"])
            self.assertFalse(counter["Q"])

        """
        P  |  P   not P
        ---+-----------
        T  |   T     F   *
        F  |   F
        """

        ante = {Var("P")}
        cons = {Not(Var("P"))}

        try:
            Sequent(ante, cons).prove()
        except CounterExample as counter:
            self.assertTrue(counter["P"])

        ante = {Not(Cond(Not(Var("P")),
                         Var("Q"))),
                Not(Cond(Var("R"),
                         Var("P"))),
                Or(Var("P"),
                   Var("R")),
                Not(Cond(Var("R"),
                         Var("Q")))}
        cons = {Not(Cond(Var("P"),
                         Var("Q")))}
        """
        P   Q   R  |  not ((not P) -> Q   not (R -> P)   P or R   not (R -> Q)   not (P -> Q)
        -----------+-------------------------------------------------------------------------
        T   T   T  |           F
        T   T   F  |           F
        T   F   T  |           F
        T   F   F  |           F
        F   T   T  |           F
        F   T   F  |           F
        F   F   T  |           T                 T           T           T             F       *
        F   F   F  |           T                 F
        """

        try:
            Sequent(ante, cons).prove()
        except CounterExample as counter:
            self.assertFalse(counter["P"])
            self.assertFalse(counter["Q"])
            self.assertTrue(counter["R"])
