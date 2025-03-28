{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05330d43-4357-436f-9479-8250c293aefc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from setuptools import setup, find_packages\n",
    "\n",
    "setup(\n",
    "    name='gridworld-mdp',\n",
    "    version='1.0.0',\n",
    "    packages=find_packages(),\n",
    "    install_requires=[\n",
    "        'tkinter'\n",
    "    ],\n",
    "    entry_points={\n",
    "        'console_scripts': [\n",
    "            'gridworld-mdp=src.gridworld_mdp:main',\n",
    "        ],\n",
    "    },\n",
    ")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
