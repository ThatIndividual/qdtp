from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Set, List

"""
The following program uses sequent calculus rules to check if a given argument
is valid or not. We will explain what sequents are in the terms of propositional
logic and how their rules are applied. The reason for which sequents have these
specific qualities are beyond the scope of this comment (and would be many times
longer than the program itself).

Sequent notation explainer
--------------------------

A sequent is a logical sentence of the following form:

    a1 and a2 and ... and an -> b1 or b2 or ... or bn

Meaning the following are sequents:

     P and Q -> R or S
     P and Q -> R
           P -> Q or R
           P -> Q
       non P -> (non P) or (P and Q)
    (P or Q) -> (P and Q)

The notation for sequents comes with a shorthand. Instead of writing

    P and Q and R -> S or T
    
we write

    {P, Q, R} => {S, T}
    
It can happen that either side of the sequent is empty, for example, the
following are properly formed sequents:

    {P, non P} => {}
            {} => {P -> (Q -> P)}

How would we translate such sequents back into common logical notation? Should
the left side be empty, it is the same thing as it being equal to true. Should
the right side be empty, it is the same thing as it being equal false. Meaning
the above can be translated as such:

    P and (non P) -> false
             true -> (P -> (Q -> P))


Sequent rule notation explainer
-------------------------------

The sequent rules use a line to separate the premises from the conclusion, like
so:

     a     b
    --------
     a and b
     
A rule of this form simply means "if we know that "a" is true and we know that
"b" is true, then we can deduce that "a and b" is true". Usually the rules go
from simple statements on the top to more complex statements on the bottom.

Because the program does deduction (taking a complex sentence and trying to find
values for the basic propositions that disprove it) and not computation (given
the values of the basic propositions, what is the value of a given complex
sentence), we actually apply the rules "in reverse".

So the above translation becomes: "if "a and b" is to be false, then either "a"
is false or "b" is false".

Of course, our rules pertain to sequents, so they look more like the following:

              L => R + {a}
    ----------------------
    {not a} + L => R
    
This is the rule for negation elimination on the left side.

"L" and "R" are simply sets that represent what other statements might be on the
left or right side. They can be empty. In plain language the above rule is
"given a sequent that has "not a" on the left side, we can transform it into a
sequent that has "a" on the right side". For example, the following is a correct
application:

           {(not a) -> (not b)} => {not b, a}
    ---------------------------------------------
    {not a, (not a) -> (not b)} => {not b}

An aditional example, conditional elimination on the left side:

    L => R + {a}   {b} + L => R
    ---------------------------
        {a -> b} + L => R
        
Here, the rule branches, just like in our initial example. If the bottom is to
be false then one of these branches must also be false. And in order to find out
which we must of course exhaust both of them. Let's apply this rule to our
earlier application:

            {} => {not b, a, not a}    {not b} => {not b, a}
            ------------------------------------------------
               {(not a) -> (not b)} => {not b, a}
    ---------------------------------------------
    {not a, (not a) -> (not b)} => {not b}

These rules applications go on until a branch has only a sequent containing
basic propositions on both sides. This sequent can be simplified no further and
at this point we know if we have a counterexample on our hands.

If the two sides share at least one element, then the sequent cannot be made to
be false no matter what values we assign to our propositions. For example:

    {P, Q} => {Q, R}   translates to   (P and Q) -> (Q or R)
    
Which is a tautology, no matter what values we assign to P, Q, R, it will always
be true (sketch a truth table to convince yourself of this). If this is the case
we say that the branch is "closed" and return to any other branch that can be
furthered simplified.

If every branch closes, then the initial argument is valid, we have failed to
find any kind of counterexample to disprove it.

But what if we find a sequent whose sides have nothing in common? We can easily
derive a counterexample from it thusly:
    
    - everything on the left side is assigned true
    - everything on the right side is assigned false
    
So, for example, if we end in the following branch:

    {P, Q} => {R}   we have the counterexample   P: true
                                                 Q: true
                                                 R: false

and we can stop applying any further rules as on counterexample is enough for
the given argument to be proven invalid.
"""


class Sentence(metaclass=ABCMeta):
    pass


@dataclass(eq=True, frozen=True)
class Var(Sentence):
    symbol: str


class ComplexSentence(Sentence, metaclass=ABCMeta):
    left_branching = False
    right_branching = False

    @abstractmethod
    def left_rule(self):
        raise NotImplementedError()

    @abstractmethod
    def right_rule(self):
        raise NotImplementedError()


@dataclass(eq=True, frozen=True)
class Not(ComplexSentence):
    negand: Sentence

    left_branching = False
    right_branching = False

    def left_rule(self):
        """
                  L => R + {a}
        ----------------------
        {not a} + L => R
        """
        return set(), {self.negand}

    def right_rule(self):
        """
        {not a} + L => R
        ----------------------
                  L => R + {a}
        """
        return {self.negand}, set()


@dataclass(eq=True, frozen=True)
class Or(ComplexSentence):
    left: Sentence
    right: Sentence

    left_branching = True
    right_branching = False

    def left_rule(self):
        """
        {a} + L => R   {b} + L => R
        ---------------------------
             {a or b} + L => R
        """
        return ({self.left}, set()), ({self.right}, set())

    def right_rule(self):
        """
        L => R + {a, b}
        -----------------
        L => R + {a or b}
        """
        return set(), {self.left, self.right}


@dataclass(eq=True, frozen=True)
class And(ComplexSentence):
    left: Sentence
    right: Sentence

    left_branching = False
    right_branching = True

    def left_rule(self):
        """
           {a, b} + L => R
        ------------------
        {a and b} + L => R
        """
        return {self.left, self.right}, set()

    def right_rule(self):
        """
        L => R + {a}   L => R + {b}
        ---------------------------
             L => R + {a and b}
        """
        return (set(), {self.left}), (set(), {self.right})


@dataclass(eq=True, frozen=True)
class Cond(ComplexSentence):
    ante: Sentence
    cons: Sentence

    left_branching = True
    right_branching = False

    def left_rule(self):
        """
        L => R + {a}   {b} + L => R
        ---------------------------
            {a -> b} + L => R
        """
        return (set(), {self.ante}), ({self.cons}, set())

    def right_rule(self):
        """
        {a} + L => R + {b}
        -----------------------
              L => R + {a -> b}
        """
        return {self.ante}, {self.cons}


class CounterExample(Exception):
    def __getitem__(self, item):
        return self.args[0][item]


@dataclass(eq=True, frozen=True)
class RuleApp:
    sequent: "Sequent"
    children: List["RuleApp"]


@dataclass(eq=True, frozen=True)
class Sequent:
    ante: Set[Sentence]
    cons: Set[Sentence]

    @classmethod
    def from_left_app(cls, sentence, old_ante, old_cons, new_ante, new_cons):
        ante = (old_ante - {sentence}) | new_ante
        cons = old_cons | new_cons
        return Sequent(ante, cons)

    @classmethod
    def from_right_app(cls, sentence, old_ante, old_cons, new_ante, new_cons):
        ante = old_ante | new_ante
        cons = (old_cons - {sentence}) | new_cons
        return Sequent(ante, cons)

    def prove(self):
        # First off we try to solve all the non-branching connectives
        for sentence in self.ante:
            if isinstance(sentence, ComplexSentence) and not sentence.left_branching:
                new_sequent = Sequent.from_left_app(sentence,
                                                    self.ante, self.cons,
                                                    *sentence.left_rule())
                new_proof = new_sequent.prove()
                return RuleApp(self, [new_proof])
        for sentence in self.cons:
            if isinstance(sentence, ComplexSentence) and not sentence.right_branching:
                new_sequent = Sequent.from_right_app(sentence,
                                                     self.ante, self.cons,
                                                     *sentence.right_rule())
                new_proof = new_sequent.prove()
                return RuleApp(self, [new_proof])

        # There are no non-branching rules left, time to branch
        for sentence in self.ante:
            if isinstance(sentence, ComplexSentence) and sentence.left_branching:
                res_left, res_right = sentence.left_rule()

                left_sequent = Sequent.from_left_app(sentence,
                                                     self.ante, self.cons,
                                                     *res_left)
                left_proof = left_sequent.prove()

                right_sequent = Sequent.from_left_app(sentence,
                                                      self.ante, self.cons,
                                                      *res_right)
                right_proof = right_sequent.prove()

                return RuleApp(self, [left_proof, right_proof])
        for sentence in self.cons:
            if isinstance(sentence, ComplexSentence) and sentence.right_branching:
                res_left, res_right = sentence.right_rule()

                left_sequent = Sequent.from_right_app(sentence,
                                                      self.ante, self.cons,
                                                      *res_left)
                left_proof = left_sequent.prove()

                right_sequent = Sequent.from_right_app(sentence,
                                                       self.ante, self.cons,
                                                       *res_right)
                right_proof = right_sequent.prove()

                return RuleApp(self, [left_proof, right_proof])

        # All the sentences we have left are atomic. We can apply the thinning
        # rule to either close this branch off or obtain a counterexample.
        #
        # Thinning:
        #     L => R
        # --------------
        # K + L => R + S
        #
        # Axiom:
        #
        # ------
        # L => L

        common = self.ante & self.cons
        if len(common):
            if len(self.ante) > 1 or len(self.cons) > 1:
                eve_prop = list(common)[0]
                axiom = Sequent({eve_prop}, {eve_prop})
                return RuleApp(self, [RuleApp(axiom, [])])
            else:
                return RuleApp(self, [])
        else:
            counter = {}
            for proposition in self.ante:
                counter[proposition.symbol] = True
            for proposition in self.cons:
                counter[proposition.symbol] = False
            raise CounterExample(counter)
