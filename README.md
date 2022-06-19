<B>Projekt Zaliczeniowy <BR></B>
Michał Szopiński <BR>
Jan Michalski <BR>
Celem zadania projektowego było znalezienie najlepszego modelu predykcji dla otrzymanych danych.

Otrzymaliśmy je w trzech plikach z tego jeden z nich zawierał dane testowe, pozostałe zaś dzieliły się na zbiór danych i odpowiadających im etykiet składających się z dwóch klas (-1,1). Całość data setu składała się 10 tysięcy kolumn i 3750 rekordów, a całość została przez nas zaklasyfikowana jako problem klasyfikacji. 
Posiadane przez nas dane sprawdziliśmy pod względem rozkładu normalnego. Tylko dwie kolumny wartości nie dopasowały się do tego rozkładu. Następnie zindefikowaliśmy 17928 outlier’ow, zamieniliśmy je na puste wartości które później zastąpiliśmy medianą z kolumny. Ostatecznie dokonaliśmy redukcji wykorzystując KPCA. Wykonaliśmy też wizualizacje danych za pomocą Tsne, oraz PCA+Tsne(ze standaryzacją i bez standaryzacji). Po przetestowaniu tych scenariuszy nie zdecydowaliśmy się na standaryzację danych ponieważ dane po standaryzacji miały bardziej charakter rozproszony co mogłoby utrudnić modelowi porpawną klasyfikację danych

W kolejnym kroku zdefiniowaliśmy naszego baseline’a (DummyClasifier’a), korzystając ze strategi najczęściej występującego wyniku,z naszego Dummiego wyznaczyliśmy takie cechy charakterystyczne:

F1 Score: 0.9413092550790069

Confusion matrix przedstawia się następująco:

![image](https://user-images.githubusercontent.com/101052451/174487836-191365bb-6cbe-42eb-be4b-040b73eabcfe.png)

Do wyznaczenia najlepszego modelu użyliśmy GridSearch/ Randomized GridSearch oraz H2O(AutoML). Przy GridSearch’u najlepszym modelem okazał się KNeighborsClassifier. Z następującymi wynikami:

F1 Score: 0.9747504403992954
Confusion matrix:
  
![image](https://user-images.githubusercontent.com/101052451/174487930-84047b23-6388-45a6-b786-eaefd795bb3b.png)

Jak widać, wykorzystany przez nas algorytm nie tylko poprawił swoja predykcję co widać w różnicy między wynikami F1,ale jednocześnie większość wyników wcześniej zaklasyfikowanych jako fałszywie pozytywne, została poprawnie oznaczona jako prawdziwie negatywne.
Wartym zaznaczenia jest fakt, że H2O skończył z minimalnie lepszym wynikiem dla F1 Score, jednak ilość fałszywie negatywnych wyników była zdecydowanie większa dlatego wierzymy że model KNN jest lepszy jakościowo od modelu wyznaczonego przez H2O

F1 score dla h2o: 0.9786003470213996

confusion matrix dla h2o
![image](https://user-images.githubusercontent.com/101052451/174488410-ba86263b-2a6d-4e9b-bb68-efc92554e0fb.png)

Podsumowując. 
Udowodniliśmy, że do posiadanego przez nas problemu klasyfikacyjnego najlepszym modelem do predykcji danych jest wykorzystanie modelu KNeighborsClassifier z następującymi hiper parametrami:
  
  ![image](https://user-images.githubusercontent.com/101052451/174488480-54de845c-39bd-48ad-9597-007aa25362b9.png)

, co zapewni nam relatywnie wysoki poziom predykcji przy zachowaniu mniejszego wyniku fałszywie negatywnych odpowiedzi w 95%
