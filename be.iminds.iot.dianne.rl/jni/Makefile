MY_TARGETS = ALE

.PHONY: install clean cleanall $(MY_TARGETS)
install clean cleanall: $(MY_TARGETS)

$(MY_TARGETS):
	$(MAKE) -C $@ $(MAKECMDGOALS);