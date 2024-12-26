import pytest

@pytest.fixture
def att():
    x = 1
    y = 2
    print("returned")
    return x + y
    
def test_a(att):
    c = att
    print("c", c)

if __name__ == "__main__":
    pytest.main()