# FAQ Search Engine using NLP

## Current Approach:

The search engine looks to be performing fairly decent with word matching on the Title and content of the help page.
The Help pages and Articles are distributed across various sections thus search functionality is definetely required.

## Flaws:

- Generic words are given equal weightage in the search
- Typos are not getting sorted.
- Context Understanding capability

Example 1:

> > Search results for <B>need motgage</B>
> >
> > - Do you need an estate plan?
> > - Do I need to renew my connection?
> > - What you need to know

Example 2:

> > Search results for <B>loan for a property</B>
> >
> > - Loan calculator
> > - Personal loan
> > - Understanding loan rates

## Solution

No point re-inventing any existing capability in search engine, its quite good if we manage to fix some evident flaws, it should be good enough for a search engine.

Quick solution to this can be a quick python pipeline between User input and the actual backend API for the seach. Idea here is to covert the user input into a piece of text that can be easily looked up by the existing solution.

Need to fix Spell Checker First >> Done using Spell Checker
Need to impement stopword removal >> Just use Pre-trained model from hugging face.

Search Results post application of this solution:

Example 1:

> > Search results for <B>need motgage</B>
> >
> > - How do I change my mortgage Direct Debit details?
> > - How to get a mortgage
> > - How do I make an overpayment to my mortgage?

Example 1 with Auto-Correct:

> > Search results for <B>need motgage</B>
> >
> > - How to get a mortgage
> > - What’s a mortgage reserve?
> > - How do I pay off my mortgage?

Example 2:

> > Search results for <B>loan for a property</B>
> >
> > - How to get a mortgage
> > - How much can I borrow for my mortgage?
> > - What’s a mortgage reserve?

## Future Scope

Here comes the actual Gen AI capability to build conversation chat bots for end users.

Additional resources worth checking out:

- ProsusAI/finbert
- PolyAI/banking77
