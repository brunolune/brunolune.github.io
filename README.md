## Apprendre l'apprentissage renforcé avec le Pacman d'UC Berkeley

<p align="center">
  <img src="http://ai.berkeley.edu/images/pacman_game.gif" alt="pacman_gif"/>
</p>

L'ambition de ce projet était de se familiariser avec les méthodes du reinforcement learning (RL) en entrainant un "agent" à jouer à un jeu simple.
On a utilisé le livre de Sutton et Barto, "Reinforcement Learning: an introduction" (http://incompleteideas.net/book/bookdraft2018jan1.pdf), pour s'initier à la théorie et aux méthodes de RL.
Notre choix de jeu s'est arrêté sur le Pacman du cours "Intro to AI (CS188)" de UC Berkeley (http://ai.berkeley.edu/reinforcement.html) pour ses qualités pédagogiques: 
- le développement des programmes et algorithmes de la partie reinforcement learning suit bien l'élaboration des concepts et méthodes du livre de Barto et Sutton.
- les programmes sont en python en style objet orienté: La structure hiérarchique des classes reflètent l'organisation et l'interdépendence des concepts dans la théorie.
- le language python permet d'utiliser les packages keras et tensorflow pour la création de réseaux de neurones



L'Objectif: entraîner un agent à evoluer de façon optimale dans un environnment.

Comment? La méthode consiste à attribuer à chaque état du système Agent-Environnement une valeur qui correspond à l'espérance du score que l'agent peut atteindre dans le futur à partir de cet état.


La familiarisation aux concepts du RL se fait en plusieurs étapes avec le cours de UC Berkeley. Dans une première étape, on se familiarise avec les concepts de Markov Decision Process (MDP) et Dynamic Programming (DP). Ces méthodes sont utiles lorsqu'on a une parfaite connaissance du système. Le MDP conceptualise l'interaction d'un agent avec son environnement. L'interaction agent-environnement est entièrement caracterisée par les transitions successives entre états (S=State), suivant des choix d'actions (A=Action) de l'agent et résultant en des bonus ou malus pour l'agent (R=reward): cela donne lieu à des séries S0,A0,R1,S1,A1 ... En outre, le MDP suppose l'existence de probabilités de transition entre les états p(S',R|S,A) qui sont connues lorsqu'on a une parfaite connaissance de l'environnement. Le MDP comprend aussi la definition d'un facteur de devaluation limitant l'importance des recompenses obtenues dans un futur lointain. Dernier point important: la valeur V(s) d'un état du systeme ne prend en compte que l'etat present et futur du systeme (the future is independent of the past given the present: c'est l'idee qu'on accumule l'experience dans la valeur de l'etat ie de la qvalues.

![Principe](https://cdn-images-1.medium.com/max/1600/1*Z2yMvuQ1-t5Ol1ac_W4dOQ.png "Principe")

Dans Pacman, les états du systèmes sont définis par la connaissance des positions de Pacman dans le labyrinthe, les positions des fantômes, et des pastilles. Les actions sont les mouvements dans les 4 directions haut, bas, gauche, droite. Pacman reçoit un bonus de:
- +10 quand il mange une pastille,
- +200 quand il mange un fantome après avoir ingéré une pastille magique,
- +500 quand il gagne après avoir mange toutes les pastilles.
Pacman recoit un malus de:
- -1 à chaque pas,
- -500 quand il se fait manger par un fantôme










<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"  
  "http://www.w3.org/TR/html4/loose.dtd">  
<html > 
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1"> 
<meta name="generator" content="TeX4ht (http://www.cse.ohio-state.edu/~gurari/TeX4ht/)"> 
<meta name="originator" content="TeX4ht (http://www.cse.ohio-state.edu/~gurari/TeX4ht/)"> 
<!-- html --> 
<meta name="src" content="testequ.tex"> 
<meta name="date" content="2018-06-14 15:49:00"> 
<link rel="stylesheet" type="text/css" href="testequ.css"> 
</head><body 
>
<!--l. 3--><p class="noindent" ><span 
class="cmmi-10">V</span> <sub><span 
class="cmmi-7">&#x03C0;</span></sub>(<span 
class="cmmi-10">s</span>) = <span 
class="cmmi-10">E</span><sub><span 
class="cmmi-7">&#x03C0;</span></sub><img 
src="testequ0x.png" alt="[Gt|St = s]"  class="left" align="middle">  
</body></html> 


You can use the [editor on GitHub](https://github.com/brunolune/brunolune.github.io/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/brunolune/brunolune.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
