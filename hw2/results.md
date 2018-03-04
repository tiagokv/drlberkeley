Answers to PDF file:

1.
 
- Which gradient estimator has better performance without advantage-centering the trajectory-centric one, or the one using reward-to-go?
  - Comparing both loss and average return over iterations, using reward-to-go leads to a faster convergence and a better converged value.
- Did advantage centering help?
  - Advantage centering does not help to converge faster or better, however diminishes the noise in the results with respect to the seed value
- Describe what you expected from the math - do the empirical results match the theory?
  - Yes, differently from supervised learning, the gradients are more unstable (probably due to variance) and the convergence takes longer. Centering the advances makes the loss more consistent as there are not values too different in their magnitude
- Did the batch size make an impact?
  - Yes, a larger batch shows less variance in the gradients, which in the loss is represented by the noise in the loss steps by iteration. It's even better with RTG
2.
  - python train_pg.py InvertedPendulum-v1 -n 100 -b 1024 -e 5 -rtg -lr 1e-2 --exp_name sb_rtg_na
3. 
  - python train_pg.py HalfCheetah-v1 -ep 150 --discount 0.9 -rtg -lr 3e-2 -l 3 -s 16 -b 4096 -e 3 -bl

Additional Questions:
- Multi step gradient: The PG converged a little bit faster, but it was much more sensitive to the learning rate used, if a lot of steps are used, sometimes even diverged
- Multi-threads generation approach: A maximum speed-up of 25% was gained parallelizing the generation in two or more threads.