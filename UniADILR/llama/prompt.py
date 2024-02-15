'''
看下只被展示不同的类别，是否会导致acc下降
(1) 0-shot 完全不展示
(2) 1-shot: 每种类别展示一个，3种类型都展示
(3) 展示1个 但是只属于某一种类别：
    当只展示 a 的时候，在 a,b,d上分别测，看下准确率
    当只展示 b 的时候，在 a,b,d 上分别测，看下准确率
    当只展示 b 的时候，在 a,b,d 上分别测，看下准确率
'''


select_prompt = 'Given the following context, please select two logical supporting facts to prove the hypothesis\
Output the number of sentences you think it is possible to prove the hypothesis,Example: Related = Sentence 3,Sentence 6\
$Now answer the question$:\n $To prove hypothesis$: {test_hypothesis} \n $context$: {test_context_text}\n'

# Example: Related = [sent3,sent6]
zero_shot_prompt = 'Please select the logical supporting facts from the context to prove the hypothesis and output a proof. Additionally, output the type of logical relationship employed in proving the hypothesis, choosing from abduction, deduction, and induction. \n\
The output should be in the following format and don\'t output any additional text otherwise:\n\
$Output Example$:\n Proof =  sent[number1] & sent[number2] -> {test_hypothesis} Reasoning Type = abduction[choose one from abduction, deduction, and induction]\n\
$Now answer the question$:\n $To prove hypothesis$: {test_hypothesis} \n $context$: {test_context_text}\n'
        


single_shot_prompt = 'Select the logical supporting facts from the context to prove the hypothesis and output a proof. \
        Additionally, output the type of logical relationship employed in proving the hypothesis, \
        choosing from abduction, deduction, and induction.\nAn Example is shown:\n"hypothesis": {example_hypothesis}\n \
        "context": {example_context_text}\n Output: The output should be in the following format and don\'t output any additional text otherwise:\n \
        "Proof = {example_proof} \nReasoning Type = {example_reasoning_type}\
        Now answer the question:\n "hypothesis": {test_hypothesis} \n "context": {test_context_text}'


# test的时候需要调整

all_prompt = 'Select the logical supporting facts from the context to prove the hypothesis and output a proof. \
        Additionally, output the type of logical relationship employed in proving the hypothesis, \
        choosing from abduction, deduction, and induction.\nSome Examples are shown:\n\
        Example1:\n \
        "hypothesis": {a_hypothesis}\n"context": {a_context_text}\n Output: \n "proof": {a_proof} \n "reasoning type": {a_reasoning_type}\n\
        Example2:\n\
        "hypothesis": {d_hypothesis}\n"context": {d_context_text}\n Output: \n "proof": {d_proof} \n "reasoning type": {d_reasoning_type}\n\
        Example3:\n \
        "hypothesis": {i_hypothesis}\n"context": {i_context_text}\n Output: \n "proof": {i_proof} \n "reasoning type": {i_reasoning_type}\n\
        Now answer the question:\n "hypothesis": {test_hypothesis} \n "context": {test_context_text}'


zero_prompt_with_descript = 'Select the logical supporting facts from the context to prove the hypothesis and output a proof. \
        Additionally, output the type of logical relationship employed in proving the hypothesis, \
        choosing from abduction, deduction, and induction. \n\
        Abductive reasoning refers to inferring that certain facts may have occurred based on observed phenomena and rules; \
        deductive reasoning refers to inferring conclusions based on rules and known premises, \
        and inductive reasoning refers to inferring general rules based on premises and results.\
        Question:\n "hypothesis": {test_hypothesis} \n "context": {test_context_text}\n\
        The output should be in the following format and don\'t output any additional text otherwise:\n\
        "Proof =  sent[m] & sent[n] -> {test_hypothesis}\n\
        Reasoning Type = abduction[choose one from abduction, deduction, and induction]"'


revise_prompt = 'Reorganize your output in the format of:\n\
        "Proof =  \nReasoning Type = "\n \
        and don\'t output no other explanations.'







zero_prompt_13b = "Given the context below, filter related sentences and provide a proof for the hypothesis '{test_hypothesis}'.\n\
Additionally, specify the type of logical relationship involved in proving the hypothesis, choosing from abduction, deduction, and induction.\n\
Generate an output in the following format:\n\
Proof = [filtered related sentences their numbers ] -> [hypothesis]\n\
Reasoning Type = [selected reasoning type]\n \
Hypothesis to Prove: {test_hypothesis} \n\
Context: \n\
{test_context_text}\n"

# sent1: If something is amenable, aggressive, and mean, then it is also angry.\
# sent2: Max is a delpee or a lompee.\
# sent3: Each twimpee is a shimpee.\
# sent4: All things that are metallic, are luminous, and are wooden, are also liquid.\
# sent5: Each folpee is a serpee.\
# sent6: Every twimpee is red.\
# sent7: Every delpee is a lompee.\
# sent8: Shimpee that are discordant, are loud, and are melodic, are muffled.\
# sent9: Max is a delpee.\
# sent10: Max is a small rimpee.\
# sent11: Max is a serpee."