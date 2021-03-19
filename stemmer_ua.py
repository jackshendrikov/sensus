import re


class UAStemmer:

    REFLEXIVE = r'(с[иья])$'  # reflexive verb
    ADJECTIVE = r'(ими|ій|ий|а|е|ова|ове|ів|є|їй|єє|еє|я|ім|ем|им|ім|их|іх|ою|йми|іми|у|ю|ого|ому|ої)$'  # adjective
    PARTICIPLE = r'(ий|ого|ому|им|ім|а|ій|у|ою|ій|і|их|йми|их)$'  # participle
    VERB = r'(сь|ся|ив|ать|ять|у|ю|ав|али|учи|ячи|вши|ши|е|ме|ати|яти|є)$'  # verb
    NOUN = r'(а|ев|ов|е|ями|ами|еи|и|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я|і|ові|ї|' \
           r'ею|єю|ою|є|еві|ем|єм|ів|їв|ю)$'  # noun

    RVRE = r'[аеиіоуяюєї]'  # vowels
    PERFECTIVE_GERUND = r'(ив|ивши|ившись|ыв|ывши|ывшись((?<=[ая])(в|вши|вшись)))$'
    DERIVATIONAL = r'[^аеиоуюяіїє][аеиоуюяіїє]+[^аеиоуюяіїє]+[аеиоуюяіїє].*(?<=о)сть?$'

    RV = ''  # the area of the word after the first vowel. It can be empty if there are no vowels in the word

    @staticmethod
    def s(s, reg, to):
        orig = s
        RV = re.sub(reg, to, s)
        return orig != RV

    @staticmethod
    def stem(word):
        word = word.lower().replace("'", "")

        if not re.search(UAStemmer.RVRE, word):
            stem = word
        else:
            p = re.search(UAStemmer.RVRE, word)
            start = word[0:p.span()[1]]
            UAStemmer.RV = word[p.span()[1]:]

            # Step 1

            # Find the end of PERFECTIVE_GERUND. If it exists, delete it and complete this step.
            #
            # Otherwise, remove the REFLEXIVE ending (if it exists). Then, in the following order,
            # we try to remove the endings: ADJECTIVAL (ADJECTIVE | PARTICIPLE + ADJECTIVE) VERB, NOUN.
            # As soon as one of them is found, the step ends.
            if not s(UAStemmer.RV, UAStemmer.PERFECTIVE_GERUND, ''):

                s(UAStemmer.RV, UAStemmer.REFLEXIVE, '')
                if s(UAStemmer.RV, UAStemmer.ADJECTIVE, ''):
                    s(UAStemmer.RV, UAStemmer.PARTICIPLE, '')
                else:
                    if not s(UAStemmer.RV, UAStemmer.VERB, ''):
                        s(UAStemmer.RV, UAStemmer.NOUN, '')
            # Step 2
            # If the word ends in 'и', delete 'и'.
            s(UAStemmer.RV, 'и$', '')

            # Step 3
            # If there is a DERIVATIONAL ending in RV, delete it.
            if re.search(UAStemmer.DERIVATIONAL, UAStemmer.RV):
                s(UAStemmer.RV, 'ость$', '')

            # Step 4
            # One of three options is possible:
            #   - If the word ends in 'ь', delete it.
            #   - If the word ends in 'нн', delete the last letter.
            #   - If the word ends in 'ейше', delete it.
            if s(UAStemmer.RV, 'ь$', ''):
                s(UAStemmer.RV, 'нн$', u'н')
                s(UAStemmer.RV, 'ейше?$', '')

            stem = start + UAStemmer.RV
        return stem
