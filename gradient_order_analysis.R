xs = c(20, 40, 60, 80, 100, 200, 300, 500, 1000, 2000, 4000)
ys = c(2.99, 5.77, 8.6, 11.4, 14.5, 28.8, 44.5, 72.8, 178, 286, 572)
x2s = xs^2

plot(xs, ys)

mod2 = lm(ys ~ xs + x2s)
summary(mod2)

mod1 = lm(ys ~ xs)
summary(mod1)
anova(mod1, mod2)
