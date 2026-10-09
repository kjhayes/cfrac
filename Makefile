
CC := gcc
CFLAGS += -O3 -ffast-math -fopenmp

EXTRA_LIB += -lncurses

default:
	@

SRC_DIR ?= ./
SRCS += \
		$(SRC_DIR)/cfrac.c

OBJS = $(SRCS:.c=.o)

%.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

cfrac: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(EXTRA_LIB)


default: cfrac

clean: FORCE
	find . -name "*.o" -delete
	rm cfrac

FORCE:

