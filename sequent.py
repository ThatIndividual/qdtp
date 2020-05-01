from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Set, List


def class_name(object):
    return type(object).__name__.lower()


class Sentence(metaclass=ABCMeta):
    @abstractmethod
    def to_latex(self):
        raise NotImplementedError()


@dataclass(eq=True, frozen=True)
class Variable(Sentence):
    symbol: str

    def to_latex(self):
        return self.symbol


class ComplexSentence(Sentence, metaclass=ABCMeta):
    left_branching = False
    right_branching = False
    symbol = None

    @abstractmethod
    def left_rule(self):
        raise NotImplementedError()

    @abstractmethod
    def right_rule(self):
        raise NotImplementedError()


@dataclass(eq=True, frozen=True)
class Negation(ComplexSentence):
    negand: Sentence

    left_branching = False
    right_branching = False
    symbol = "\\neg"

    def to_latex(self):
        return f"\\neg {self.negand.to_latex()}"

    def left_rule(self):
        return set(), {self.negand}

    def right_rule(self):
        return {self.negand}, set()


@dataclass(eq=True, frozen=True)
class Disjunction(ComplexSentence):
    left: Sentence
    right: Sentence

    left_branching = True
    right_branching = False
    symbol = "\\vee"

    def to_latex(self):
        return f"({self.left.to_latex()} \\vee {self.right.to_latex()})"

    def left_rule(self):
        return ({self.left}, set()), ({self.right}, set())

    def right_rule(self):
        return set(), {self.left, self.right}


@dataclass(eq=True, frozen=True)
class Conjunction(ComplexSentence):
    left: Sentence
    right: Sentence

    left_branching = False
    right_branching = True
    symbol = "\\wedge"

    def to_latex(self):
        return f"({self.left.to_latex()} \\wedge {self.right.to_latex()})"

    def left_rule(self):
        return {self.left, self.right}, set()

    def right(self):
        return (set(), {self.left}), (set(), {self.right})


@dataclass(eq=True, frozen=True)
class Conditional(ComplexSentence):
    antecedent: Sentence
    consequent: Sentence

    left_branching = True
    right_branching = False
    symbol = "\\rightarrow"

    def to_latex(self):
        return f"({self.antecedent.to_latex()} \\rightarrow {self.consequent.to_latex()})"

    def left_rule(self):
        return (set(), {self.antecedent}), ({self.consequent}, set())

    def right_rule(self):
        return {self.antecedent}, {self.consequent}


class CounterExample(Exception):
    pass


@dataclass(eq=True, frozen=True)
class RuleApp:
    sequent: "Sequent"
    children: List["RuleApp"]
    rule: str

    def to_latex(self):
        if self.rule == "counter":
            return f"\\hypo{{ {self.sequent.to_latex()} }}\n" + \
                   "\\rewrite{\\color{red}\\box\\treebox}"
        else:
            return "\n".join(child.to_latex() for child in self.children) + \
                   f"\n\\infer{len(self.children)}[{self.rule}]{{ {self.sequent.to_latex()} }}"


@dataclass(eq=True, frozen=True)
class Sequent:
    antecedents: Set[Sentence]
    consequents: Set[Sentence]

    def to_latex(self):
        left = ", ".join(antecedent.to_latex() for antecedent in self.antecedents) or "Ø"
        right = ", ".join(consequent.to_latex() for consequent in self.consequents) or "Ø"
        return f"{left} &\\Rightarrow {right}"

    @classmethod
    def from_left_app(cls, sentence, old_ante, old_cons, new_ante, new_cons):
        antecedents = (old_ante - {sentence}) | new_ante
        consequents = old_cons | new_cons
        return Sequent(antecedents, consequents)

    @classmethod
    def from_right_app(cls, sentence, old_ante, old_cons, new_ante, new_cons):
        antecedents = old_ante | new_ante
        consequents = (old_cons - {sentence}) | new_cons
        return Sequent(antecedents, consequents)

    def prove(self):
        # First off we try to solve all the non-branching connectives
        for sentence in self.antecedents:
            if isinstance(sentence, ComplexSentence) and not sentence.left_branching:
                new_sequent = Sequent.from_left_app(sentence,
                                                    self.antecedents, self.consequents,
                                                    *sentence.left_rule())
                new_proof = new_sequent.prove()
                return RuleApp(self, [new_proof], f"left {sentence.symbol}")
        for sentence in self.consequents:
            if isinstance(sentence, ComplexSentence) and not sentence.right_branching:
                new_sequent = Sequent.from_right_app(sentence,
                                                     self.antecedents, self.consequents,
                                                     *sentence.right_rule())
                new_proof = new_sequent.prove()
                return RuleApp(self, [new_proof], f"right {sentence.symbol}")

        # There are no non-branching rules left, time to branch
        for sentence in self.antecedents:
            if isinstance(sentence, ComplexSentence) and sentence.left_branching:
                res_left, res_right = sentence.left_rule()

                left_sequent = Sequent.from_left_app(sentence,
                                                     self.antecedents, self.consequents,
                                                     *res_left)
                left_proof = left_sequent.prove()

                right_sequent = Sequent.from_left_app(sentence,
                                                      self.antecedents, self.consequents,
                                                      *res_right)
                right_proof = right_sequent.prove()

                return RuleApp(self, [left_proof, right_proof], f"left {sentence.symbol}")
        for sentence in self.consequents:
            if isinstance(sentence, ComplexSentence) and sentence.right_branching:
                res_left, res_right = sentence.right_rule()

                left_sequent = Sequent.from_right_app(sentence,
                                                      self.antecedents, self.consequents,
                                                      *res_left)
                left_proof = left_sequent.prove()

                right_sequent = Sequent.from_right_app(sentence,
                                                       self.antecedents, self.consequents,
                                                       *res_right)
                right_proof = right_sequent.prove()

                return RuleApp(self, [left_proof, right_proof], f"right {sentence.symbol}")

        # All the sentences we have left are atomic. We can apply the thinning
        # rule to either close this branch off or obtain a counterexample.
        common = self.antecedents & self.consequents
        if len(common):
            if len(self.antecedents) > 1 or len(self.consequents) > 1:
                eve_prop = list(common)[0]
                axiom = Sequent({eve_prop}, {eve_prop})
                return RuleApp(self, [RuleApp(axiom, [], "axiom")], "thinning")
            else:
                return RuleApp(self, [], "axiom")
        else:
            return RuleApp(self, [], "counter")


if __name__ == "__main__":
    ante = {Conditional(Disjunction(Variable("A"), Negation(Variable("B"))), Variable("C")),
            Conditional(Variable("B"), Negation(Variable("D"))),
            Variable("D")}
    cons = {Variable("C")}
    # ante = {Variable("Q"),
    #         Conditional(Variable("P"), Variable("Q"))}
    # cons = {Variable("P")}
    # ante = set()
    # cons = {Conditional(Variable("P"), Conditional(Variable("Q"), Variable("P")))}
    # ante = {Variable("P")}
    # cons = {Negation(Negation(Variable("P")))}
    # ante = {Variable("P"),
    #         Conditional(Variable("P"), Variable("Q")),
    #         Conditional(Variable("Q"), Variable("R"))}
    # cons = {Variable("R")}
    sequent = Sequent(ante, cons)
    proof = sequent.prove()
    print(proof.to_latex())
    pass
