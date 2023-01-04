# Model

## Load model
```python
model = torch.load(PATH.pb)
```

## Load from model parameter
```python
model = Net()
model.load_state_dict(torch.load(PATH.pt))
model.eval()
```