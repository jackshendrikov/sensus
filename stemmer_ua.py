import re


class UAStemmer:

    def __init__(self, word):
        self.word = word

        self.RVRE = r'[аеиіоуяюєї]'  # vowels
        self.REFLEXIVE = r'(с[иья])$'  # reflexive verb
        self.ADJECTIVE = r'(ими|ій|ий|а|е|ова|ове|ів|є|їй|єє|еє|я|ім|ем|им|ім|их|іх|ою|йми|іми|' \
                         r'у|ю|ого|ому|ої)$'  # adjective
        self.PARTICIPLE = r'(ий|ого|ому|им|ім|а|ій|у|ою|ій|і|их|йми|их)$'  # participle
        self.VERB = r'(сь|ся|ив|ать|ять|у|ю|ав|али|учи|ячи|вши|ши|е|ме|ати|яти|є)$'  # verb
        self.NOUN = r'(а|ев|ов|е|ями|ами|еи|и|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я|і|' \
                    r'ові|ї|ею|єю|ою|є|еві|ем|єм|ів|їв|ю)$'  # noun
        self.PERFECTIVE_GERUND = r'(ив|ивши|ившись|ыв|ывши|ывшись((?<=[ая])(в|вши|вшись)))$'
        self.DERIVATIONAL = r'[^аеиоуюяіїє][аеиоуюяіїє]+[^аеиоуюяіїє]+[аеиоуюяіїє].*(?<=о)сть?$'

        self.RV = ''  # the area of the word after the first vowel. It can be empty if there are no vowels in the word

    def s(self, s, reg, to):
        orig = s
        RV = re.sub(reg, to, s)
        return orig != RV

    def stem_word(self):
        word = self.word.lower().replace("'", "")

        if not re.search(self.RVRE, word):
            stem = word
        else:
            p = re.search(self.RVRE, word)
            start = word[0:p.span()[1]]
            self.RV = word[p.span()[1]:]

            # Step 1

            # Find the end of PERFECTIVE_GERUND. If it exists, delete it and complete this step.
            #
            # Otherwise, remove the REFLEXIVE ending (if it exists). Then, in the following order,
            # we try to remove the endings: ADJECTIVAL (ADJECTIVE | PARTICIPLE + ADJECTIVE) VERB, NOUN.
            # As soon as one of them is found, the step ends.
            if not self.s(self.RV, self.PERFECTIVE_GERUND, ''):

                self.s(self.RV, self.REFLEXIVE, '')
                if self.s(self.RV, self.ADJECTIVE, ''):
                    self.s(self.RV, self.PARTICIPLE, '')
                else:
                    if not self.s(self.RV, self.VERB, ''):
                        self.s(self.RV, self.NOUN, '')
            # Step 2
            # If the word ends in 'и', delete 'и'.
            self.s(self.RV, 'и$', '')

            # Step 3
            # If there is a DERIVATIONAL ending in RV, delete it.
            if re.search(self.DERIVATIONAL, self.RV):
                self.s(self.RV, 'ость$', '')

            # Step 4
            # One of three options is possible:
            #   - If the word ends in 'ь', delete it.
            #   - If the word ends in 'нн', delete the last letter.
            #   - If the word ends in 'ейше', delete it.
            if self.s(self.RV, 'ь$', ''):
                self.s(self.RV, 'нн$', u'н')
                self.s(self.RV, 'ейше?$', '')

            stem = start + self.RV
        return stem
