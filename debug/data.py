from ..experiment import DataLoader


def test_multi():
    print()
    dl = DataLoader(set_name='mult')
    content = list(dl.load_set())
    print(content[0])


def test_volkova():
    print()
    dl = DataLoader(set_name='volk')
    content = list(dl.load_set())
    print(content[0])
