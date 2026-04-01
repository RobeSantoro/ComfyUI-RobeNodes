import { app } from "/scripts/app.js";

const EXTENSION_ID = "Robe.normalizePath";
const MENU_LABEL = "normalize path";

app.registerExtension({
	name: EXTENSION_ID,
	getCanvasMenuItems() {
		const autoTarget = detectCurrentPlatformPathStyle();

		return [
			null,
			{
				content: MENU_LABEL,
				has_submenu: true,
				submenu: {
					options: [
						{
							content: `Auto (${formatTargetLabel(autoTarget)})`,
							callback: () => normalizeWorkflowPaths(autoTarget),
						},
						{
							content: "To Windows (\\)",
							callback: () => normalizeWorkflowPaths("windows"),
						},
						{
							content: "To Unix (/)",
							callback: () => normalizeWorkflowPaths("unix"),
						},
					],
				},
			},
		];
	},
});

function normalizeWorkflowPaths(targetStyle) {
	const nodes = app.graph?._nodes ?? [];
	if (!nodes.length) {
		showToast("info", "Normalize Path", "No workflow is currently loaded.");
		return;
	}

	let changedValues = 0;
	const changedNodes = new Set();

	for (const node of nodes) {
		const widgets = Array.isArray(node.widgets) ? node.widgets : [];
		for (let index = 0; index < widgets.length; index += 1) {
			const widget = widgets[index];
			const currentValue = getWidgetValue(node, widget, index);

			if (!isStringEntryWidget(widget, currentValue)) {
				continue;
			}

			const normalizedValue = normalizeStringValue(widget?.name, currentValue, targetStyle);
			if (normalizedValue === currentValue) {
				continue;
			}

			setWidgetValue(node, widget, index, normalizedValue);
			changedValues += 1;
			changedNodes.add(node.id);
		}
	}

	if (!changedValues) {
		showToast(
			"info",
			"Normalize Path",
			`No path-like widget values needed changes for ${formatTargetLabel(targetStyle)}.`,
		);
		return;
	}

	app.graph.setDirtyCanvas(true, true);

	showToast(
		"success",
		"Normalize Path",
		`Updated ${changedValues} value${changedValues === 1 ? "" : "s"} across ${changedNodes.size} node${changedNodes.size === 1 ? "" : "s"} to ${formatTargetLabel(targetStyle)}.`,
	);
}

function getWidgetValue(node, widget, index) {
	if (widget && typeof widget.value === "string") {
		return widget.value;
	}

	if (Array.isArray(node.widgets_values)) {
		return node.widgets_values[index];
	}

	if (
		node.widgets_values &&
		typeof node.widgets_values === "object" &&
		widget?.name in node.widgets_values
	) {
		return node.widgets_values[widget.name];
	}

	return undefined;
}

function setWidgetValue(node, widget, index, value) {
	if (widget) {
		widget.value = value;
	}

	if (Array.isArray(node.widgets_values)) {
		node.widgets_values[index] = value;
	} else if (
		node.widgets_values &&
		typeof node.widgets_values === "object" &&
		widget?.name in node.widgets_values
	) {
		node.widgets_values[widget.name] = value;
	}

	node.setDirtyCanvas?.(true, true);
}

function isStringEntryWidget(widget, value) {
	if (typeof value !== "string" || !value.trim()) {
		return false;
	}

	return true;
}

function normalizeStringValue(widgetName, value, targetStyle) {
	if (value.includes("\n")) {
		const lines = value.split(/\r?\n/);
		const nonEmptyLines = lines.filter((line) => line.trim().length > 0);
		if (!nonEmptyLines.length || !nonEmptyLines.every((line) => isLikelyPathString(widgetName, line))) {
			return value;
		}

		return lines.map((line) => normalizeSeparators(line, targetStyle)).join("\n");
	}

	if (!isLikelyPathString(widgetName, value)) {
		return value;
	}

	return normalizeSeparators(value, targetStyle);
}

function isLikelyPathString(widgetName, value) {
	const trimmedValue = value.trim();
	if (!trimmedValue || !/[\\/]/.test(trimmedValue)) {
		return false;
	}

	if (isUrlLike(trimmedValue) || /\s[\\/]|[\\/]\s/.test(trimmedValue)) {
		return false;
	}

	if (
		/^[A-Za-z]:[\\/]/.test(trimmedValue) ||
		/^\\\\/.test(trimmedValue) ||
		/^(?:\/|~\/|\.\/|\.\.\/)/.test(trimmedValue)
	) {
		return true;
	}

	const widgetLabel = `${widgetName ?? ""}`.toLowerCase();
	if (/(?:path|paths|dir|directory|folder|file|filename|prefix|video|image|audio)/.test(widgetLabel)) {
		return true;
	}

	const parts = trimmedValue.split(/[\\/]+/).filter(Boolean);
	if (parts.length >= 3) {
		return true;
	}

	if (parts.length < 2) {
		return false;
	}

	const lastPart = parts[parts.length - 1];
	if (/\.[A-Za-z0-9]{1,8}$/.test(lastPart)) {
		return true;
	}

	return parts.some((part) => /[_\-.0-9 ]/.test(part));
}

function isUrlLike(value) {
	return /^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(value) || /^data:/i.test(value);
}

function normalizeSeparators(value, targetStyle) {
	if (targetStyle === "windows") {
		return value.replace(/\//g, "\\");
	}

	return value.replace(/\\/g, "/");
}

function detectCurrentPlatformPathStyle() {
	const platform = `${navigator.userAgentData?.platform ?? navigator.platform ?? navigator.userAgent ?? ""}`.toLowerCase();
	return platform.includes("win") ? "windows" : "unix";
}

function formatTargetLabel(targetStyle) {
	return targetStyle === "windows" ? "Windows (\\)" : "Unix (/)";
}

function showToast(severity, summary, detail) {
	app.extensionManager.toast.add({
		severity,
		summary,
		detail,
		life: 3500,
	});
}
