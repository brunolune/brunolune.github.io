## Apprendre l'apprentissage renforcé avec le Pacman d'UC Berkeley

<p align="center">
  <img src="http://ai.berkeley.edu/images/pacman_game.gif" alt="pacman_gif"/>
</p>

L'ambition de ce projet était de se familiariser avec les méthodes du reinforcement learning (RL) en entrainant un "agent" à jouer à un jeu simple.
On a utilisé le livre de Sutton et Barto, "Reinforcement Learning: an introduction" (http://incompleteideas.net/book/bookdraft2018jan1.pdf), pour s'initier à la théorie et aux méthodes de RL.
Notre choix de jeu s'est arrêté sur le Pacman du cours "Intro to AI (CS188)" de UC Berkeley (http://ai.berkeley.edu/reinforcement.html) pour ses qualités pédagogiques: 
- le développement des programmes et algorithmes de la partie reinforcement learning suit bien l'élaboration des concepts et méthodes du livre de Barto et Sutton.
- les programmes sont en python en style objet orienté: La structure hiérarchique des classes reflètent l'organisation et l'interdépendence des concepts dans la théorie.
- le language python permet d'utiliser les packages keras et tensorflow pour la création de réseaux de neurones.

La familiarisation aux concepts du RL se fait en plusieurs étapes avec le cours de UC Berkeley. Dans une première étape, on se familiarise avec les concepts de Markov Decision Process (MDP) et Dynamic Programming (DP). Ces méthodes sont utiles lorsqu'on a une parfaite connaissance du système. Le MDP conceptualise l'interaction d'un agent avec son environnement. L'interaction agent-environnement est entièrement caractérisée par les transitions successives entre états (S=State), suivant des choix d'actions (A=Action) de l'agent et résultant en des bonus ou malus pour l'agent (R=Reward): cela donne lieu à des séries S0,A0,R1,S1,A1 ... En outre, le MDP suppose l'existence de probabilités de transition entre les états p(S',R|S,A) et comprend aussi la définition d'un facteur de dévaluation limitant l'importance des récompenses (Reward) obtenues dans un futur lointain. Dernier point important: la valeur V(S) d'un état du système ne prend en compte que l'état présent et futur du système ("the future is independent of the past given the present").Les experiences du passe sont accumule l'experience dans la valeur de l'etat ie de la qvalues.

![Principe](https://cdn-images-1.medium.com/max/1600/1*Z2yMvuQ1-t5Ol1ac_W4dOQ.png "Principe")

Dans Pacman, les états du systèmes sont définis par la connaissance des positions de Pacman dans le labyrinthe, les positions des fantômes, et des pastilles. Les actions sont les mouvements dans les 4 directions haut, bas, gauche, droite, ou l'inaction. Les bonus et malus de Pacman sont:
- +10 quand il mange une pastille,
- +200 quand il mange un fantôme après avoir ingéré une pastille magique,
- +500 quand il gagne après avoir mange toutes les pastilles,
- -1 à chaque pas,
- -500 quand il se fait manger par un fantôme.

Objectif des méthodes de Reinforcement Learning: entraîner un agent à evoluer de façon optimale dans un environnment. Dans le cadre d'un jeu comme pacman, le comportement optimal consiste à gagner la partie en obtenant le score maximal.

Comment? La méthode consiste à attribuer à chaque état du système Agent-Environnement une valeur qui correspond à l'espérance du score que l'agent (pacman) peut atteindre dans le futur à partir de cet état S, choisisant l'action A suivant la règle de conduite (policy en anglais) dénotée par pi.

<img src="equ_1_Q.png">

Ainsi, une règle de conduite dite "avide" (greedy policy en anglais) consistera à systématiquement choisir l'action qui conduira à l'espérance de récompense maximum dans le futur immédiat.

<img src="equ_2_pi.png">

La première partie du cours de UC Berkeley est dediée à l'implémentation de l'algorithme d'itération de la valeur des états dans le cadre d'un environnement connu: les probabilités de transitions entre états sont connues et la méthode de Dynamic Programming consiste à faire converger la fonction Q(S,A) vers sa vraie valeur par un processus itératif. Une iteration passe en revue la totalité des états et actualise la valeur de Q(S,A) à partir des valeurs des proches voisins. La valeur de Q s'ajuste de proche en proche à partir des valeurs connues (terminal states) au fur et à mesure des itérations. La convergence est rapide sur des petite grilles d'états. Cette méthode devient trop couteuse en terme de puissance de calcul lorsque le nombre d'états augmente.

<img src="equ_3_Q.png">

A chaque iteration, apres avoir actualise la function Q(S,A), on actualise la regle de conduite, le tout formant un processus appele iterative policy evaluation.

<img src="equ_3_pi.png">

Interet: planifier la meilleure stategie lorsque l'environnement est connue.

L'exemple suivant est interessant pour comprendre l'effet des parametres gamma, ainsi que le role des probabilites de transitions. Les probabilites de transition reflete la capacite a effectivement controler son evolution dans l'environnement. Le parametre noise permet d'introduire un caractere aleatoire dans l'evolution d'un hypothetique agent. Par exmple sur cet exemple il pourrait s'agir d'une personne qui passe le long d'un precipice pour recuperer un tresor. Le parametre noise pourrait refleter son etat d'ebriete. Si la personne a bu, elle pourrait ne pas parfaitement controler ses mouvements ainsi elle n'aurait par exemple qu'une probabilite p(S+N,-1|S,N)=1-noise de se diriger vers le Nord (N) et p(S+X,-1|S,N)=noise/3 de se diriger vers une autre direction (X) malgre son intention d'aller vers le Nord. Le parametre gamma reflete la capacite de la personne a connaitre l'environnement lointain. Dans cet exemple, la personne pourrait choisir de recolter le tresor de moindre valeur si le parametre gamma est trop eleve etant donne que la valeur de direction vers l'etat du tresor de plus grande valeur aura ete trop diminuee.


Sur cet exemple, discount=0.9, noise=0.2, la route optimale devient le passage loin du precipice pour recuperer le tresor de plus grande valeur car le risque de tomber est trop grand, et le coefficient de devaluation (discount) n'est pas assez grand pour ne pas "voir" le tresor de plus grande valeur.



<img src="discountgrid.png">

<img src="qvalueiteration.png">












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
