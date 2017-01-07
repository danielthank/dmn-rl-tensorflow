import os
import re

from data_helper.data_utils import DataSet, WordTable

SPLIT_RE = re.compile('(\W+)?')
def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    ret = []
    for token in re.split(SPLIT_RE, sentence):
        token = token.strip()
        if token:
            ret.append(token.lower())
    return ret

def parse_task(lines, only_supporting=False):
    """
    Parse the bAbI task format described here: https://research.facebook.com/research/babi/
    If only_supporting is True, only the sentences that support the answer are kept.
    """
    stories = []
    story = []
    for line in lines:
        # line = line.decode('utf-8').strip()
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            query, answer, supporting = line.split('\t')
            query = tokenize(query)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            stories.append((substory, query, answer))
            story.append('')
        else:
            sentence = tokenize(line)
            story.append(sentence)
    return stories

def get_tokenizer(stories, word_table):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for story, query, answer in stories:
        tokens_all.extend([token for sentence in story for token in sentence] + query + [answer])
    word_table.add_vocab(*tokens_all)

def pad_task(task, max_story_length, max_sentence_length, max_query_length):
    """
    Pad sentences, stories, and queries to a consistence length.
    """
    stories = []
    questions = []
    answers = []
    for story, query, answer in task:
        for sentence in story:
            for _ in range(max_sentence_length - len(sentence)):
                sentence.append(0)
            assert len(sentence) == max_sentence_length

        for _ in range(max_story_length - len(story)):
            story.append([0 for _ in range(max_sentence_length)])

        for _ in range(max_query_length - len(query)):
            query.append(0)

        stories.append(story)
        questions.append(query)
        answers.append(answer)

        assert len(story) == max_story_length
        assert len(query) == max_query_length

    return stories, questions, answers

def truncate_task(stories, max_length):
    stories_truncated = []
    for story, query, answer in stories:
        story_truncated = story[-max_length:]
        stories_truncated.append((story_truncated, query, answer))
    return stories_truncated

def tokenize_task(stories, word_table):
    """
    Convert all tokens into their unique ids.
    """
    story_ids = []
    for story, query, answer in stories:
        story = [[word_table.word2idx[token] for token in sentence] for sentence in story]
        query = [word_table.word2idx[token] for token in query]
        answer = word_table.word2idx[answer]
        story_ids.append((story, query, answer))
    return story_ids

def read_babi(task_id, batch_size):
    """ Reads bAbi data set.
    :param task_id: task no. (int)
    :param batch_size: how many examples in a minibatch?
    """
    if task_id == 3:
        truncate_length = 130
    else:
        truncate_length = 70

    word_table = WordTable()

    f_train = open(os.path.join('babi', 'train', 'task_{}.txt'.format(task_id)))
    train = parse_task(f_train.readlines())
    train = truncate_task(train, truncate_length)
    get_tokenizer(train, word_table)
    train = tokenize_task(train, word_table)

    f_test = open(os.path.join('babi', 'test', 'task_{}.txt'.format(task_id)))
    test = parse_task(f_test.readlines())
    test = truncate_task(test, truncate_length)
    get_tokenizer(test, word_table)
    test = tokenize_task(test, word_table)

    a = max([len(story) for story, _, _ in train])
    b = max([len(story) for story, _, _ in test])
    max_story_length = max(a, b)

    a = max([len(sentence) for story, _, _ in train for sentence in story])
    b = max([len(sentence) for story, _, _ in test for sentence in story])
    max_sentence_length = max(a, b)

    a = max([len(question) for _, question, _ in train])
    b = max([len(question) for _, question, _ in test])
    max_question_length = max(a, b)

    train_stories, train_questions, train_answers = pad_task(train, max_story_length, max_sentence_length, max_question_length)
    test_stories, test_questions, test_answers = pad_task(test, max_story_length, max_sentence_length, max_question_length)

    return DataSet(batch_size, train_stories, train_questions, train_answers, name='train'), DataSet(batch_size, test_stories, test_questions, test_answers, name='test'), word_table, max_story_length, max_sentence_length, max_question_length
