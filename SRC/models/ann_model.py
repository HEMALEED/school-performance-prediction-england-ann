{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "05663966",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model successfully created!\n",
      "MLP(\n",
      "  (fc1): Linear(in_features=5, out_features=8, bias=True)\n",
      "  (fc2): Linear(in_features=8, out_features=9, bias=True)\n",
      "  (out): Linear(in_features=9, out_features=5, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "\n",
    "class MLP(nn.Module):\n",
    "\n",
    "    def __init__(self,\n",
    "                 in_features=5,\n",
    "                 h1=8,\n",
    "                 h2=9,\n",
    "                 out_features=5):\n",
    "\n",
    "        super().__init__()\n",
    "\n",
    "        self.fc1 = nn.Linear(in_features, h1)\n",
    "        self.fc2 = nn.Linear(h1, h2)\n",
    "        self.out = nn.Linear(h2, out_features)\n",
    "\n",
    "    def forward(self, x):\n",
    "\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = F.relu(self.fc2(x))\n",
    "        x = self.out(x)\n",
    "\n",
    "        return x\n",
    "    \n",
    "# Pick a manual_seed for reproducibility (No indentation here!)\n",
    "torch.manual_seed(42)\n",
    "# Create an instance of the MLP class\n",
    "model = MLP()\n",
    "print(\"Model successfully created!\")\n",
    "print(model)  "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
